package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
)

// Config contains all key hyperparameters.
//
// High-school view:
// - n_embd: size of each token vector ("how many features per token")
// - n_head: number of attention heads ("how many parallel focus mechanisms")
// - n_layer: number of stacked transformer blocks
// - block_size: maximum sequence length processed in one pass
// - learning_rate: step size for optimization
type Config struct {
	NEmpd        int     `json:"n_embd"`
	NHead        int     `json:"n_head"`
	NLayer       int     `json:"n_layer"`
	BlockSize    int     `json:"block_size"`
	LearningRate float64 `json:"learning_rate"`
}

// Model stores all trainable parameters and runtime state.
//
// Notes:
// - Params is a flat list so optimizer updates are easy.
// - State keeps matrices by readable names (simple for learning/debugging).
// - AdamM and AdamV store Adam optimizer moving averages.
// - mu protects model parameters from concurrent HTTP requests.
type Model struct {
	Config    Config
	VocabSize int
	Chars     []string
	BOS       int
	Params    []*Value
	State     map[string][][]*Value
	AdamM     []float64
	AdamV     []float64
	Steps     int
	mu        sync.Mutex
}

// NewModel builds vocabulary and initializes all transformer weights.
//
// Vocabulary setup:
// - We collect every unique rune from docs.
// - We sort characters for deterministic token IDs.
// - We append one special control token used as both BOS and END.
func NewModel(config Config, docs []string) *Model {
	charSet := make(map[rune]bool)
	for _, doc := range docs {
		for _, r := range doc {
			charSet[r] = true
		}
	}

	chars := make([]string, 0, len(charSet))
	for r := range charSet {
		chars = append(chars, string(r))
	}
	sort.Strings(chars)

	vocabSize := len(chars) + 1
	bos := len(chars)

	m := &Model{
		Config:    config,
		VocabSize: vocabSize,
		Chars:     chars,
		BOS:       bos,
		State:     make(map[string][][]*Value),
	}

	// Helper that creates a matrix and also registers each value into Params.
	createMatrix := func(rows, cols int) [][]*Value {
		mat := make([][]*Value, rows)
		for i := 0; i < rows; i++ {
			mat[i] = make([]*Value, cols)
			for j := 0; j < cols; j++ {
				// Small Gaussian initialization keeps activations stable initially.
				val := NewValue(rand.NormFloat64() * 0.02)
				mat[i][j] = val
				m.Params = append(m.Params, val)
			}
		}
		return mat
	}

	// Token embedding, position embedding, and output projection.
	m.State["wte"] = createMatrix(vocabSize, config.NEmpd)
	m.State["wpe"] = createMatrix(config.BlockSize, config.NEmpd)
	m.State["lm_head"] = createMatrix(vocabSize, config.NEmpd)

	// Per-layer matrices for attention and MLP.
	for i := 0; i < config.NLayer; i++ {
		m.State[fmt.Sprintf("layer%d.attn_wq", i)] = createMatrix(config.NEmpd, config.NEmpd)
		m.State[fmt.Sprintf("layer%d.attn_wk", i)] = createMatrix(config.NEmpd, config.NEmpd)
		m.State[fmt.Sprintf("layer%d.attn_wv", i)] = createMatrix(config.NEmpd, config.NEmpd)
		m.State[fmt.Sprintf("layer%d.attn_wo", i)] = createMatrix(config.NEmpd, config.NEmpd)
		m.State[fmt.Sprintf("layer%d.mlp_fc1", i)] = createMatrix(4*config.NEmpd, config.NEmpd)
		m.State[fmt.Sprintf("layer%d.mlp_fc2", i)] = createMatrix(config.NEmpd, 4*config.NEmpd)
	}

	m.AdamM = make([]float64, len(m.Params))
	m.AdamV = make([]float64, len(m.Params))

	return m
}

// Linear computes y = W*x where:
// - x is a vector
// - W is matrix with shape [out_dim][in_dim]
func (m *Model) Linear(x []*Value, w [][]*Value) []*Value {
	out := make([]*Value, len(w))
	for i, row := range w {
		sum := NewValue(0)
		for j, xi := range x {
			sum = sum.Add(row[j].Mul(xi))
		}
		out[i] = sum
	}
	return out
}

// Softmax converts logits into probabilities that sum to 1.
//
// We subtract max logit first for numerical stability.
func (m *Model) Softmax(logits []*Value) []*Value {
	maxVal := -math.MaxFloat64
	for _, l := range logits {
		if l.Data > maxVal {
			maxVal = l.Data
		}
	}

	exps := make([]*Value, len(logits))
	total := NewValue(0)
	for i, l := range logits {
		e := l.Add(NewValue(-maxVal)).Exp()
		exps[i] = e
		total = total.Add(e)
	}

	invTotal := total.Pow(-1)
	probs := make([]*Value, len(logits))
	for i, e := range exps {
		probs[i] = e.Mul(invTotal)
	}
	return probs
}

// RMSNorm normalizes vector magnitude using root-mean-square.
// This keeps scale stable and helps training.
func (m *Model) RMSNorm(x []*Value) []*Value {
	sumSq := NewValue(0)
	for _, xi := range x {
		sumSq = sumSq.Add(xi.Mul(xi))
	}
	ms := sumSq.Mul(NewValue(1.0 / float64(len(x))))
	scale := ms.Add(NewValue(1e-5)).Pow(-0.5)

	out := make([]*Value, len(x))
	for i, xi := range x {
		out[i] = xi.Mul(scale)
	}
	return out
}

// Update performs one Adam optimization step over all parameters.
func (m *Model) Update() {
	m.Steps++

	lr := m.Config.LearningRate
	beta1, beta2, eps := 0.85, 0.99, 1e-8

	for i, p := range m.Params {
		m.AdamM[i] = beta1*m.AdamM[i] + (1-beta1)*p.Grad
		m.AdamV[i] = beta2*m.AdamV[i] + (1-beta2)*p.Grad*p.Grad

		// Bias-corrected first and second moments.
		mHat := m.AdamM[i] / (1 - math.Pow(beta1, float64(m.Steps)))
		vHat := m.AdamV[i] / (1 - math.Pow(beta2, float64(m.Steps)))

		p.Data -= lr * mHat / (math.Sqrt(vHat) + eps)
		p.Grad = 0
	}
}
