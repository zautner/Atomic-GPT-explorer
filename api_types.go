package main

// InitRequest is the payload for /api/init.
// It provides training docs and model hyperparameters.
type InitRequest struct {
	Docs   []string `json:"docs"`
	Config Config   `json:"config"`
}

// TrainResponse reports one training step summary.
type TrainResponse struct {
	Step          int     `json:"step"`
	Loss          float64 `json:"loss"`
	ContextChar   string  `json:"context_char"`
	TargetChar    string  `json:"target_char"`
	PredictedChar string  `json:"predicted_char"`
	TargetProb    float64 `json:"target_prob"`
	PredictedProb float64 `json:"predicted_prob"`
}

// TraceCandidate is one candidate token shown in generation trace.
type TraceCandidate struct {
	Char    string  `json:"char"`
	TokenID int     `json:"token_id"`
	Logit   float64 `json:"logit"`
	Prob    float64 `json:"prob"`
}

// TraceStep explains one sampled generation position.
type TraceStep struct {
	Position   int              `json:"position"`
	Context    string           `json:"context"`
	TopK       []TraceCandidate `json:"top_k"`
	RandomU    float64          `json:"random_u"`
	ChosenChar string           `json:"chosen_char"`
	ChosenProb float64          `json:"chosen_prob"`
	ChosenRank int              `json:"chosen_rank"`
	CumBefore  float64          `json:"cum_before"`
	CumAfter   float64          `json:"cum_after"`
	Reason     string           `json:"reason"`
}

// GenerateTraceResponse is returned by /api/generate_trace.
type GenerateTraceResponse struct {
	Text       string      `json:"text"`
	Steps      []TraceStep `json:"steps"`
	StopReason string      `json:"stop_reason"`
}
