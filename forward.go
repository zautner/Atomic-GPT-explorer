package main

import (
	"fmt"
	"math"
)

// Forward runs one autoregressive step:
// it consumes a single token + position and returns logits for next token.
//
// keys/values are KV caches, one per layer, that store past sequence state.
// This allows current token to attend to earlier tokens.
func (m *Model) Forward(tokenID, posID int, keys, values [][][]*Value) []*Value {
	// Embed token and position, then add them.
	tokEmb := m.State["wte"][tokenID]
	posEmb := m.State["wpe"][posID]
	x := make([]*Value, m.Config.NEmpd)
	for i := 0; i < m.Config.NEmpd; i++ {
		x[i] = tokEmb[i].Add(posEmb[i])
	}
	x = m.RMSNorm(x)

	headDim := m.Config.NEmpd / m.Config.NHead

	// Process each transformer layer.
	for li := 0; li < m.Config.NLayer; li++ {
		// -------- Attention block --------
		xResidual := x
		x = m.RMSNorm(x)

		q := m.Linear(x, m.State[fmt.Sprintf("layer%d.attn_wq", li)])
		k := m.Linear(x, m.State[fmt.Sprintf("layer%d.attn_wk", li)])
		v := m.Linear(x, m.State[fmt.Sprintf("layer%d.attn_wv", li)])
		keys[li] = append(keys[li], k)
		values[li] = append(values[li], v)

		xAttn := make([]*Value, 0, m.Config.NEmpd)

		// Multi-head attention:
		// each head looks at a slice of embedding dimensions.
		for h := 0; h < m.Config.NHead; h++ {
			hs := h * headDim
			qH := q[hs : hs+headDim]

			// Score each past position with q·k/sqrt(d).
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

			// Weighted sum of value vectors.
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

		// Project concatenated heads back to embedding size + residual connection.
		x = m.Linear(xAttn, m.State[fmt.Sprintf("layer%d.attn_wo", li)])
		for i := range x {
			x[i] = x[i].Add(xResidual[i])
		}

		// -------- MLP block --------
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

	// Convert final hidden vector to logits over vocabulary.
	return m.Linear(x, m.State["lm_head"])
}
