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

// TrainRequest controls how much work /api/train performs in one call.
//
// All fields are optional; server uses safe defaults when omitted.
type TrainRequest struct {
	StepsPerCall int `json:"steps_per_call"`
	BatchSize    int `json:"batch_size"`
}

// GenerateOptions controls stochastic sampling behavior.
//
// Temperature:
// - < 1.0 => sharper/more deterministic output
// - > 1.0 => more random output
//
// TopK:
// - 0 => disabled
// - N => keep only N highest-probability tokens before sampling
//
// MinLen:
// - minimum characters to emit before allowing <END>.
type GenerateOptions struct {
	Temperature float64 `json:"temperature"`
	TopK        int     `json:"top_k"`
	MinLen      int     `json:"min_len"`
}

// GenerateRequest allows options for /api/generate and /api/generate_trace.
type GenerateRequest struct {
	Options GenerateOptions `json:"options"`
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
