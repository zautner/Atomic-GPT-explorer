# Atomic GPT Explorer (Pure Go)

Single Go application with:
- Go backend API
- Go-served static frontend (no Node.js, no Vite, no TypeScript)
- Small educational character-level transformer/autograd implementation

Important scope note:
- This is a toy model trained from scratch on user-provided strings at runtime.
- It is not a pretrained production GPT model.

## Requirements

- Go 1.21+

## Run

```bash
go run .
```

Open:
- http://127.0.0.1:8080/
- Docs page: http://127.0.0.1:8080/docs/

Live website:
- http://atomicgpt.duckdns.org:8080/

## Build

```bash
go build ./...
```

## Project layout

- `main.go`: app bootstrap (embed assets, register routes, start server)
- `server.go`: HTTP handlers and shared server state
- `api_types.go`: request/response structs for API
- `autograd.go`: tiny autodiff engine (`Value`, ops, `Backward`)
- `model.go`: model config/state, initialization, math helpers, optimizer
- `forward.go`: transformer forward pass
- `inference_and_training.go`: training step + sampling + trace generation
- `web/index.html`: main UI
- `web/app.js`: browser logic
- `web/docs/index.html`: help page

## API endpoints

1. `POST /api/init`
- Purpose: initialize model with documents and hyperparameters.
- Body:
```json
{
  "docs": ["alex", "anna", "john"],
  "config": {
    "n_embd": 16,
    "n_head": 4,
    "n_layer": 1,
    "block_size": 16,
    "learning_rate": 0.05
  }
}
```

2. `POST /api/train`
- Purpose: train model parameters.
- Body is optional.
- Optional body fields:
```json
{
  "steps_per_call": 1,
  "batch_size": 6
}
```
- Defaults when omitted:
- `steps_per_call = 1`
- `batch_size = 6`

3. `POST /api/generate`
- Purpose: sample generated text.
- Body is optional.
- Optional body fields:
```json
{
  "options": {
    "temperature": 0.7,
    "top_k": 5,
    "min_len": 3
  }
}
```
- Defaults when omitted:
- `temperature = 0.7`
- `top_k = 5`
- `min_len = 3`

4. `POST /api/generate_trace`
- Purpose: sample generated text and return per-step sampling trace.
- Accepts the same optional `options` as `/api/generate`.
