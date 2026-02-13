package main

import (
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"sync"
)

// Server owns HTTP handlers and shared application state.
//
// Why separate this from Model?
// - Model is "ML math + parameters."
// - Server is "request handling + lifecycle/state wiring."
type Server struct {
	mu    sync.RWMutex
	model *Model
	docs  []string
}

// NewServer creates an empty API server.
func NewServer() *Server {
	return &Server{}
}

// RegisterRoutes attaches all endpoints to the provided mux.
func (s *Server) RegisterRoutes(mux *http.ServeMux, webRoot fs.FS) {
	mux.HandleFunc("/api/init", s.handleInit)
	mux.HandleFunc("/api/train", s.handleTrain)
	mux.HandleFunc("/api/generate", s.handleGenerate)
	mux.HandleFunc("/api/generate_trace", s.handleGenerateTrace)
	mux.Handle("/", http.FileServer(http.FS(webRoot)))
}

// snapshot reads current model/docs atomically with shared lock.
func (s *Server) snapshot() (*Model, []string) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.model, append([]string(nil), s.docs...)
}

// setModel swaps active model/docs atomically with exclusive lock.
func (s *Server) setModel(model *Model, docs []string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.model = model
	s.docs = append([]string(nil), docs...)
}

// writeJSON is a helper to consistently send JSON responses.
func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

// decodeOptionalJSON decodes JSON when body is present.
// Empty bodies are treated as "use defaults" rather than errors.
func decodeOptionalJSON(r *http.Request, dst any) error {
	if r.Body == nil {
		return nil
	}
	err := json.NewDecoder(r.Body).Decode(dst)
	if err == io.EOF {
		return nil
	}
	return err
}

func (s *Server) handleInit(w http.ResponseWriter, r *http.Request) {
	var req InitRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	model := NewModel(req.Config, req.Docs)
	s.setModel(model, req.Docs)

	// Keep response shape compatible with existing frontend behavior.
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = fmt.Fprintf(w, `{"status":"initialized","params":%d}`, len(model.Params))
}

func (s *Server) handleTrain(w http.ResponseWriter, r *http.Request) {
	model, docs := s.snapshot()
	if model == nil {
		http.Error(w, "Model not initialized", http.StatusBadRequest)
		return
	}
	if len(docs) == 0 {
		http.Error(w, "No training documents provided", http.StatusBadRequest)
		return
	}

	// Lock model during forward/backward/update to avoid concurrent mutation.
	model.mu.Lock()
	defer model.mu.Unlock()

	req := TrainRequest{}
	if err := decodeOptionalJSON(r, &req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	stepsPerCall := req.StepsPerCall
	if stepsPerCall <= 0 {
		stepsPerCall = 2
	}
	batchSize := req.BatchSize
	if batchSize <= 0 {
		batchSize = 4
	}

	resp, err := TrainBatchedSteps(model, docs, stepsPerCall, batchSize)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	model, _ := s.snapshot()
	if model == nil {
		http.Error(w, "Model not initialized", http.StatusBadRequest)
		return
	}

	model.mu.Lock()
	defer model.mu.Unlock()

	req := GenerateRequest{}
	if err := decodeOptionalJSON(r, &req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	opts := req.Options
	if opts.Temperature <= 0 {
		opts.Temperature = 0.7
	}
	if opts.TopK <= 0 {
		opts.TopK = 5
	}
	if opts.MinLen <= 0 {
		opts.MinLen = 3
	}

	text := GenerateSample(model, opts)
	writeJSON(w, http.StatusOK, map[string]string{"text": text})
}

func (s *Server) handleGenerateTrace(w http.ResponseWriter, r *http.Request) {
	model, _ := s.snapshot()
	if model == nil {
		http.Error(w, "Model not initialized", http.StatusBadRequest)
		return
	}

	model.mu.Lock()
	defer model.mu.Unlock()

	req := GenerateRequest{}
	if err := decodeOptionalJSON(r, &req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	opts := req.Options
	if opts.Temperature <= 0 {
		opts.Temperature = 0.7
	}
	if opts.TopK <= 0 {
		opts.TopK = 5
	}
	if opts.MinLen <= 0 {
		opts.MinLen = 3
	}

	writeJSON(w, http.StatusOK, GenerateSampleWithTrace(model, opts))
}
