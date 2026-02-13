package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
)

// tokenLabel converts token IDs to human-readable labels.
// This project uses one shared control token for BOS and END, displayed as <END>.
func tokenLabel(tokenID, bos int, chars []string) string {
	if tokenID == bos {
		return "<END>"
	}
	return chars[tokenID]
}

// encodeDoc turns a string into token IDs, wrapped with BOS at both ends.
//
// Why both ends?
// - Starting BOS gives the model a standard "sequence starts now" signal.
// - Ending BOS plays the role of an end token for training completion.
func encodeDoc(doc string, chars []string, bos int) []int {
	tokens := []int{bos}
	for _, char := range doc {
		for idx, c := range chars {
			if c == string(char) {
				tokens = append(tokens, idx)
				break
			}
		}
	}
	tokens = append(tokens, bos)
	return tokens
}

// sampleFromProbVector picks one token using inverse transform sampling.
//
// Steps:
// 1) Draw random number u in [0,1).
// 2) Walk probabilities cumulatively until interval contains u.
// 3) Return the selected token and interval details.
func sampleFromProbVector(probs []float64, fallbackTokenID int) (chosen int, u, cumBefore, cumAfter, chosenProb float64) {
	u = rand.Float64()
	cumulative := 0.0

	chosen = fallbackTokenID
	chosenProb = 0.0
	cumBefore = 0.0
	cumAfter = 0.0

	for idx, p := range probs {
		prev := cumulative
		cumulative += p
		if u < cumulative {
			chosen = idx
			chosenProb = p
			cumBefore = prev
			cumAfter = cumulative
			return
		}
	}

	// If numerical issues prevented selection, return final interval.
	cumAfter = cumulative
	return
}

// topKCandidates selects the K highest-probability tokens for debugging display.
func topKCandidates(logits, probs []float64, chars []string, bos, k int) []TraceCandidate {
	indices := make([]int, len(probs))
	for i := range probs {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		return probs[indices[i]] > probs[indices[j]]
	})
	if len(indices) > k {
		indices = indices[:k]
	}

	out := make([]TraceCandidate, 0, len(indices))
	for _, idx := range indices {
		out = append(out, TraceCandidate{
			Char:    tokenLabel(idx, bos, chars),
			TokenID: idx,
			Logit:   logits[idx],
			Prob:    probs[idx],
		})
	}
	return out
}

// trainOneExample computes one training loss and backpropagates gradients.
//
// It does not update parameters by itself.
func trainOneExample(model *Model, docs []string) (TrainResponse, error) {
	doc := docs[rand.Intn(len(docs))]
	tokens := encodeDoc(doc, model.Chars, model.BOS)

	n := len(tokens) - 1
	if n > model.Config.BlockSize {
		n = model.Config.BlockSize
	}
	if n <= 0 {
		return TrainResponse{}, fmt.Errorf("training sequence is empty")
	}

	keys := make([][][]*Value, model.Config.NLayer)
	values := make([][][]*Value, model.Config.NLayer)
	losses := []*Value{}
	contextChar := "<END>"
	targetChar := "<END>"
	predictedChar := "<END>"
	targetProb := 0.0
	predictedProb := 0.0

	// Teacher forcing:
	// - feed current token
	// - train to predict next token
	for pos := 0; pos < n; pos++ {
		logits := model.Forward(tokens[pos], pos, keys, values)
		probs := model.Softmax(logits)
		loss := probs[tokens[pos+1]].Log().Mul(NewValue(-1))
		losses = append(losses, loss)

		// Record final position diagnostics for UI.
		if pos == n-1 {
			bestIdx := 0
			bestProb := probs[0].Data
			for idx, p := range probs {
				if p.Data > bestProb {
					bestIdx = idx
					bestProb = p.Data
				}
			}
			contextChar = tokenLabel(tokens[pos], model.BOS, model.Chars)
			targetChar = tokenLabel(tokens[pos+1], model.BOS, model.Chars)
			predictedChar = tokenLabel(bestIdx, model.BOS, model.Chars)
			targetProb = probs[tokens[pos+1]].Data
			predictedProb = bestProb
		}
	}

	// Average loss over positions.
	totalLoss := NewValue(0)
	for _, l := range losses {
		totalLoss = totalLoss.Add(l)
	}
	avgLoss := totalLoss.Mul(NewValue(1.0 / float64(n)))
	avgLoss.Backward()

	return TrainResponse{
		Step:          model.Steps,
		Loss:          avgLoss.Data,
		ContextChar:   contextChar,
		TargetChar:    targetChar,
		PredictedChar: predictedChar,
		TargetProb:    targetProb,
		PredictedProb: predictedProb,
	}, nil
}

// TrainBatchedSteps runs multiple optimizer steps, each with gradient accumulation
// over a mini-batch of random examples.
func TrainBatchedSteps(model *Model, docs []string, stepsPerCall, batchSize int) (TrainResponse, error) {
	if stepsPerCall < 1 {
		stepsPerCall = 1
	}
	if batchSize < 1 {
		batchSize = 1
	}

	lastResp := TrainResponse{}
	avgLossAcrossSteps := 0.0

	for step := 0; step < stepsPerCall; step++ {
		// Ensure gradients are clean before accumulating batch gradients.
		for _, p := range model.Params {
			p.Grad = 0
		}

		batchLoss := 0.0
		for b := 0; b < batchSize; b++ {
			docResp, err := trainOneExample(model, docs)
			if err != nil {
				return TrainResponse{}, err
			}
			batchLoss += docResp.Loss
			lastResp = docResp
		}

		// Scale gradients by batch size so update magnitude remains stable.
		scale := 1.0 / float64(batchSize)
		for _, p := range model.Params {
			p.Grad *= scale
		}

		model.Update()
		avgLossAcrossSteps += batchLoss / float64(batchSize)
	}

	lastResp.Step = model.Steps
	lastResp.Loss = avgLossAcrossSteps / float64(stepsPerCall)
	return lastResp, nil
}

// samplingConfig returns validated generation defaults/options.
func samplingConfig(opts GenerateOptions, vocabSize int) GenerateOptions {
	if opts.Temperature <= 0 {
		opts.Temperature = 0.7
	}
	if opts.TopK < 0 {
		opts.TopK = 0
	}
	if opts.TopK > vocabSize {
		opts.TopK = vocabSize
	}
	if opts.MinLen < 0 {
		opts.MinLen = 0
	}
	return opts
}

// toProbVector applies temperature, optional top-k filtering, and optional
// temporary suppression of <END>, then returns final sampling probabilities.
func toProbVector(logits []*Value, opts GenerateOptions, bosTokenID int, suppressEnd bool) ([]float64, []float64) {
	raw := make([]float64, len(logits))
	maxLogit := -math.MaxFloat64
	for i := range logits {
		raw[i] = logits[i].Data / opts.Temperature
		if raw[i] > maxLogit {
			maxLogit = raw[i]
		}
	}

	probs := make([]float64, len(raw))
	sumExp := 0.0
	for i := range raw {
		v := math.Exp(raw[i] - maxLogit)
		probs[i] = v
		sumExp += v
	}
	if sumExp > 0 {
		for i := range probs {
			probs[i] /= sumExp
		}
	}

	// Keep only top-k options if requested.
	if opts.TopK > 0 && opts.TopK < len(probs) {
		indices := make([]int, len(probs))
		for i := range probs {
			indices[i] = i
		}
		sort.Slice(indices, func(i, j int) bool {
			return probs[indices[i]] > probs[indices[j]]
		})

		mask := make([]bool, len(probs))
		for i := 0; i < opts.TopK; i++ {
			mask[indices[i]] = true
		}
		for i := range probs {
			if !mask[i] {
				probs[i] = 0
			}
		}
	}

	if suppressEnd && bosTokenID >= 0 && bosTokenID < len(probs) {
		probs[bosTokenID] = 0
	}

	// Renormalize after filtering/suppression.
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	if sum > 0 {
		for i := range probs {
			probs[i] /= sum
		}
	} else {
		// Fallback: make distribution valid even in degenerate cases.
		uniform := 1.0 / float64(len(probs))
		for i := range probs {
			probs[i] = uniform
		}
		if suppressEnd && len(probs) > 1 {
			probs[bosTokenID] = 0
			rest := 1.0 / float64(len(probs)-1)
			for i := range probs {
				if i != bosTokenID {
					probs[i] = rest
				}
			}
		}
	}

	return raw, probs
}

// GenerateSample creates one sampled text without detailed trace.
func GenerateSample(model *Model, opts GenerateOptions) string {
	opts = samplingConfig(opts, model.VocabSize)
	tokenID := model.BOS
	sample := []string{}
	keys := make([][][]*Value, model.Config.NLayer)
	values := make([][][]*Value, model.Config.NLayer)

	for pos := 0; pos < model.Config.BlockSize; pos++ {
		logits := model.Forward(tokenID, pos, keys, values)
		suppressEnd := len(sample) < opts.MinLen
		_, probs := toProbVector(logits, opts, model.BOS, suppressEnd)
		newTokenID, _, _, _, _ := sampleFromProbVector(probs, model.BOS)

		if newTokenID == model.BOS {
			break
		}

		sample = append(sample, model.Chars[newTokenID])
		tokenID = newTokenID
	}

	return strings.Join(sample, "")
}

// GenerateSampleWithTrace creates sampled text and explains each choice.
func GenerateSampleWithTrace(model *Model, opts GenerateOptions) GenerateTraceResponse {
	opts = samplingConfig(opts, model.VocabSize)
	tokenID := model.BOS
	sample := []string{}
	keys := make([][][]*Value, model.Config.NLayer)
	values := make([][][]*Value, model.Config.NLayer)
	steps := []TraceStep{}
	stopReason := "Reached block size limit"

	for pos := 0; pos < model.Config.BlockSize; pos++ {
		logits := model.Forward(tokenID, pos, keys, values)
		suppressEnd := len(sample) < opts.MinLen
		rawLogits, probs := toProbVector(logits, opts, model.BOS, suppressEnd)
		topK := topKCandidates(rawLogits, probs, model.Chars, model.BOS, 5)

		newTokenID, rnd, cumBefore, cumAfter, chosenProb := sampleFromProbVector(probs, model.BOS)

		chosenRank := len(probs)
		for rank, cand := range topK {
			if cand.TokenID == newTokenID {
				chosenRank = rank + 1
				break
			}
		}

		reason := fmt.Sprintf(
			"Chosen '%s' because draw %.4f fell inside cumulative interval [%.4f, %.4f) in vocabulary index order.",
			tokenLabel(newTokenID, model.BOS, model.Chars),
			rnd,
			cumBefore,
			cumAfter,
		)
		if len(topK) > 0 && topK[0].TokenID != newTokenID {
			reason += fmt.Sprintf(
				" Highest-probability option was '%s' at %.4f, but stochastic sampling can still pick lower-ranked valid options.",
				topK[0].Char,
				topK[0].Prob,
			)
		}

		steps = append(steps, TraceStep{
			Position:   pos,
			Context:    strings.Join(sample, ""),
			TopK:       topK,
			RandomU:    rnd,
			ChosenChar: tokenLabel(newTokenID, model.BOS, model.Chars),
			ChosenProb: chosenProb,
			ChosenRank: chosenRank,
			CumBefore:  cumBefore,
			CumAfter:   cumAfter,
			Reason:     reason,
		})

		if newTokenID == model.BOS {
			stopReason = "Model selected <END> token"
			break
		}

		sample = append(sample, model.Chars[newTokenID])
		tokenID = newTokenID
	}

	return GenerateTraceResponse{
		Text:       strings.Join(sample, ""),
		Steps:      steps,
		StopReason: stopReason,
	}
}
