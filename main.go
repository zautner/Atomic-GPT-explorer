package main

import (
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"math"
	"math/rand"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

//go:embed web/*
var webFS embed.FS

// --- Autograd Engine ---

type Value struct {
	Data       float64
	Grad       float64
	Children   []*Value
	LocalGrads []float64
}

func NewValue(data float64) *Value {
	return &Value{Data: data}
}

func (v *Value) Add(other *Value) *Value {
	return &Value{
		Data:       v.Data + other.Data,
		Children:   []*Value{v, other},
		LocalGrads: []float64{1, 1},
	}
}

func (v *Value) Mul(other *Value) *Value {
	return &Value{
		Data:       v.Data * other.Data,
		Children:   []*Value{v, other},
		LocalGrads: []float64{other.Data, v.Data},
	}
}

func (v *Value) Pow(other float64) *Value {
	return &Value{
		Data:       math.Pow(v.Data, other),
		Children:   []*Value{v},
		LocalGrads: []float64{other * math.Pow(v.Data, other-1)},
	}
}

func (v *Value) Log() *Value {
	return &Value{
		Data:       math.Log(v.Data),
		Children:   []*Value{v},
		LocalGrads: []float64{1 / v.Data},
	}
}

func (v *Value) Exp() *Value {
	exp := math.Exp(v.Data)
	return &Value{
		Data:       exp,
		Children:   []*Value{v},
		LocalGrads: []float64{exp},
	}
}

func (v *Value) Relu() *Value {
	grad := 0.0
	if v.Data > 0 {
		grad = 1.0
	}
	return &Value{
		Data:       math.Max(0, v.Data),
		Children:   []*Value{v},
		LocalGrads: []float64{grad},
	}
}

func (v *Value) Backward() {
	topo := []*Value{}
	visited := make(map[*Value]bool)
	var buildTopo func(*Value)
	buildTopo = func(node *Value) {
		if !visited[node] {
			visited[node] = true
			for _, child := range node.Children {
				buildTopo(child)
			}
			topo = append(topo, node)
		}
	}
	buildTopo(v)

	v.Grad = 1
	for i := len(topo) - 1; i >= 0; i-- {
		curr := topo[i]
		for j, child := range curr.Children {
			child.Grad += curr.LocalGrads[j] * curr.Grad
		}
	}
}

// --- GPT Model ---

type Config struct {
	NEmpd        int     `json:"n_embd"`
	NHead        int     `json:"n_head"`
	NLayer       int     `json:"n_layer"`
	BlockSize    int     `json:"block_size"`
	LearningRate float64 `json:"learning_rate"`
}

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

func NewModel(config Config, docs []string) *Model {
	charSet := make(map[rune]bool)
	for _, doc := range docs {
		for _, r := range doc {
			charSet[r] = true
		}
	}
	var chars []string
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

	createMatrix := func(rows, cols int) [][]*Value {
		mat := make([][]*Value, rows)
		for i := 0; i < rows; i++ {
			mat[i] = make([]*Value, cols)
			for j := 0; j < cols; j++ {
				val := NewValue(rand.NormFloat64() * 0.02)
				mat[i][j] = val
				m.Params = append(m.Params, val)
			}
		}
		return mat
	}

	m.State["wte"] = createMatrix(vocabSize, config.NEmpd)
	m.State["wpe"] = createMatrix(config.BlockSize, config.NEmpd)
	m.State["lm_head"] = createMatrix(vocabSize, config.NEmpd)

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

func (m *Model) Forward(tokenID, posID int, keys, values [][][]*Value) []*Value {
	tokEmb := m.State["wte"][tokenID]
	posEmb := m.State["wpe"][posID]

	x := make([]*Value, m.Config.NEmpd)
	for i := 0; i < m.Config.NEmpd; i++ {
		x[i] = tokEmb[i].Add(posEmb[i])
	}
	x = m.RMSNorm(x)

	headDim := m.Config.NEmpd / m.Config.NHead

	for li := 0; li < m.Config.NLayer; li++ {
		// Attention
		xResidual := x
		x = m.RMSNorm(x)
		q := m.Linear(x, m.State[fmt.Sprintf("layer%d.attn_wq", li)])
		k := m.Linear(x, m.State[fmt.Sprintf("layer%d.attn_wk", li)])
		v := m.Linear(x, m.State[fmt.Sprintf("layer%d.attn_wv", li)])
		keys[li] = append(keys[li], k)
		values[li] = append(values[li], v)

		xAttn := make([]*Value, 0, m.Config.NEmpd)
		for h := 0; h < m.Config.NHead; h++ {
			hs := h * headDim
			qH := q[hs : hs+headDim]

			attnLogits := make([]*Value, len(keys[li]))
			for t := 0; t < len(keys[li]); t++ {
				dot := NewValue(0)
				kH := keys[li][t][hs : hs+headDim]
				for j := 0; j < headDim; j++ {
					dot = dot.Add(qH[j].Mul(kH[j]))
				}
				attnLogits[t] = dot.Mul(NewValue(1.0 / math.Sqrt(float64(headDim))))
			}
			attnWeights := m.Softmax(attnLogits)

			headOut := make([]*Value, headDim)
			for j := 0; j < headDim; j++ {
				sum := NewValue(0)
				for t := 0; t < len(values[li]); t++ {
					vH := values[li][t][hs : hs+headDim]
					sum = sum.Add(attnWeights[t].Mul(vH[j]))
				}
				headOut[j] = sum
			}
			xAttn = append(xAttn, headOut...)
		}
		x = m.Linear(xAttn, m.State[fmt.Sprintf("layer%d.attn_wo", li)])
		for i := range x {
			x[i] = x[i].Add(xResidual[i])
		}

		// MLP
		xResidual = x
		x = m.RMSNorm(x)
		x = m.Linear(x, m.State[fmt.Sprintf("layer%d.mlp_fc1", li)])
		for i := range x {
			x[i] = x[i].Relu()
		}
		x = m.Linear(x, m.State[fmt.Sprintf("layer%d.mlp_fc2", li)])
		for i := range x {
			x[i] = x[i].Add(xResidual[i])
		}
	}

	return m.Linear(x, m.State["lm_head"])
}

func (m *Model) Update() {
	m.Steps++
	lr := m.Config.LearningRate
	beta1, beta2, eps := 0.85, 0.99, 1e-8

	for i, p := range m.Params {
		m.AdamM[i] = beta1*m.AdamM[i] + (1-beta1)*p.Grad
		m.AdamV[i] = beta2*m.AdamV[i] + (1-beta2)*p.Grad*p.Grad
		mHat := m.AdamM[i] / (1 - math.Pow(beta1, float64(m.Steps)))
		vHat := m.AdamV[i] / (1 - math.Pow(beta2, float64(m.Steps)))
		p.Data -= lr * mHat / (math.Sqrt(vHat) + eps)
		p.Grad = 0
	}
}

// --- API Server ---

var globalModel *Model
var globalDocs []string

type InitRequest struct {
	Docs   []string `json:"docs"`
	Config Config   `json:"config"`
}

type TrainResponse struct {
	Step int     `json:"step"`
	Loss float64 `json:"loss"`
}

type TraceCandidate struct {
	Char    string  `json:"char"`
	TokenID int     `json:"token_id"`
	Logit   float64 `json:"logit"`
	Prob    float64 `json:"prob"`
}

type TraceStep struct {
	Position   int              `json:"position"`
	Context    string           `json:"context"`
	TopK       []TraceCandidate `json:"top_k"`
	RandomU    float64          `json:"random_u"`
	ChosenChar string           `json:"chosen_char"`
	ChosenProb float64          `json:"chosen_prob"`
	Reason     string           `json:"reason"`
}

type GenerateTraceResponse struct {
	Text       string      `json:"text"`
	Steps      []TraceStep `json:"steps"`
	StopReason string      `json:"stop_reason"`
}

func tokenLabel(tokenID, bos int, chars []string) string {
	if tokenID == bos {
		return "<END>"
	}
	return chars[tokenID]
}

func topKCandidates(logits, probs []*Value, chars []string, bos, k int) []TraceCandidate {
	indices := make([]int, len(probs))
	for i := range probs {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		return probs[indices[i]].Data > probs[indices[j]].Data
	})
	if len(indices) > k {
		indices = indices[:k]
	}

	out := make([]TraceCandidate, 0, len(indices))
	for _, idx := range indices {
		out = append(out, TraceCandidate{
			Char:    tokenLabel(idx, bos, chars),
			TokenID: idx,
			Logit:   logits[idx].Data,
			Prob:    probs[idx].Data,
		})
	}
	return out
}

func main() {
	rand.Seed(time.Now().UnixNano())
	webRoot, err := fs.Sub(webFS, "web")
	if err != nil {
		log.Fatalf("failed to load web assets: %v", err)
	}

	http.HandleFunc("/api/init", func(w http.ResponseWriter, r *http.Request) {
		var req InitRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		globalDocs = req.Docs
		globalModel = NewModel(req.Config, globalDocs)
		w.WriteHeader(200)
		fmt.Fprintf(w, `{"status":"initialized","params":%d}`, len(globalModel.Params))
	})

	http.HandleFunc("/api/train", func(w http.ResponseWriter, r *http.Request) {
		if globalModel == nil {
			http.Error(w, "Model not initialized", 400)
			return
		}
		if len(globalDocs) == 0 {
			http.Error(w, "No training documents provided", 400)
			return
		}
		globalModel.mu.Lock()
		defer globalModel.mu.Unlock()

		doc := globalDocs[rand.Intn(len(globalDocs))]
		tokens := []int{globalModel.BOS}
		for _, char := range doc {
			for idx, c := range globalModel.Chars {
				if c == string(char) {
					tokens = append(tokens, idx)
					break
				}
			}
		}
		tokens = append(tokens, globalModel.BOS)

		n := len(tokens) - 1
		if n > globalModel.Config.BlockSize {
			n = globalModel.Config.BlockSize
		}

		keys := make([][][]*Value, globalModel.Config.NLayer)
		values := make([][][]*Value, globalModel.Config.NLayer)
		losses := []*Value{}

		for pos := 0; pos < n; pos++ {
			logits := globalModel.Forward(tokens[pos], pos, keys, values)
			probs := globalModel.Softmax(logits)
			loss := probs[tokens[pos+1]].Log().Mul(NewValue(-1))
			losses = append(losses, loss)
		}

		totalLoss := NewValue(0)
		for _, l := range losses {
			totalLoss = totalLoss.Add(l)
		}
		avgLoss := totalLoss.Mul(NewValue(1.0 / float64(n)))
		avgLoss.Backward()
		globalModel.Update()

		json.NewEncoder(w).Encode(TrainResponse{
			Step: globalModel.Steps,
			Loss: avgLoss.Data,
		})
	})

	http.HandleFunc("/api/generate", func(w http.ResponseWriter, r *http.Request) {
		if globalModel == nil {
			http.Error(w, "Model not initialized", 400)
			return
		}
		globalModel.mu.Lock()
		defer globalModel.mu.Unlock()

		tokenID := globalModel.BOS
		sample := []string{}
		keys := make([][][]*Value, globalModel.Config.NLayer)
		values := make([][][]*Value, globalModel.Config.NLayer)

		for pos := 0; pos < globalModel.Config.BlockSize; pos++ {
			logits := globalModel.Forward(tokenID, pos, keys, values)
			probs := globalModel.Softmax(logits)

			// Sampling
			rnd := rand.Float64()
			cumulative := 0.0
			newTokenID := globalModel.BOS
			for idx, p := range probs {
				cumulative += p.Data
				if rnd < cumulative {
					newTokenID = idx
					break
				}
			}

			if newTokenID == globalModel.BOS {
				break
			}
			sample = append(sample, globalModel.Chars[newTokenID])
			tokenID = newTokenID
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{
			"text": strings.Join(sample, ""),
		})
	})

	http.HandleFunc("/api/generate_trace", func(w http.ResponseWriter, r *http.Request) {
		if globalModel == nil {
			http.Error(w, "Model not initialized", 400)
			return
		}
		globalModel.mu.Lock()
		defer globalModel.mu.Unlock()

		tokenID := globalModel.BOS
		sample := []string{}
		keys := make([][][]*Value, globalModel.Config.NLayer)
		values := make([][][]*Value, globalModel.Config.NLayer)
		steps := []TraceStep{}
		stopReason := "Reached block size limit"

		for pos := 0; pos < globalModel.Config.BlockSize; pos++ {
			logits := globalModel.Forward(tokenID, pos, keys, values)
			probs := globalModel.Softmax(logits)
			topK := topKCandidates(logits, probs, globalModel.Chars, globalModel.BOS, 5)

			rnd := rand.Float64()
			cumulative := 0.0
			newTokenID := globalModel.BOS
			chosenProb := 0.0
			for idx, p := range probs {
				cumulative += p.Data
				if rnd < cumulative {
					newTokenID = idx
					chosenProb = p.Data
					break
				}
			}

			reason := fmt.Sprintf(
				"Chosen '%s' because cumulative probability %.4f crossed random draw %.4f.",
				tokenLabel(newTokenID, globalModel.BOS, globalModel.Chars),
				cumulative,
				rnd,
			)
			if len(topK) > 0 && topK[0].TokenID != newTokenID {
				reason += fmt.Sprintf(
					" Highest-probability option was '%s' at %.4f, but sampling selected a different valid token.",
					topK[0].Char,
					topK[0].Prob,
				)
			}

			steps = append(steps, TraceStep{
				Position:   pos,
				Context:    strings.Join(sample, ""),
				TopK:       topK,
				RandomU:    rnd,
				ChosenChar: tokenLabel(newTokenID, globalModel.BOS, globalModel.Chars),
				ChosenProb: chosenProb,
				Reason:     reason,
			})

			if newTokenID == globalModel.BOS {
				stopReason = "Model selected <END> token"
				break
			}
			sample = append(sample, globalModel.Chars[newTokenID])
			tokenID = newTokenID
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(GenerateTraceResponse{
			Text:       strings.Join(sample, ""),
			Steps:      steps,
			StopReason: stopReason,
		})
	})

	http.Handle("/", http.FileServer(http.FS(webRoot)))

	fmt.Println("Server starting on :8080...")
	http.ListenAndServe(":8080", nil)
}
