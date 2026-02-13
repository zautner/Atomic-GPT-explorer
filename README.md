# Atomic GPT Explorer (Pure Go)

This project now runs as a single Go application:
- Go backend API
- Go-served static frontend (no Node.js, no Vite, no TypeScript)

## Requirements

- Go 1.21+

## Run

```bash
go run main.go
```

Open:
- http://127.0.0.1:8080/

## Build

```bash
go build ./...
```

## Project layout

- `main.go`: model + API + static file serving
- `web/index.html`: main UI
- `web/app.js`: browser logic
- `web/docs/index.html`: help page
