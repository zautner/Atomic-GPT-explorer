# Atomic GPT Explorer (Pure Go)

Single Go application with:
- Go backend API
- Go-served static frontend (no Node.js, no Vite, no TypeScript)
- Small educational transformer/autograd implementation

## Requirements

- Go 1.21+

## Run

```bash
go run .
```

Open:
- http://127.0.0.1:8080/

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

- `POST /api/init`: initialize model with docs + config
- `POST /api/train`: run one training step
- `POST /api/generate`: sample generated text
- `POST /api/generate_trace`: sample with step-by-step explanation
