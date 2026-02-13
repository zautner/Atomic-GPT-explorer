package main

import (
	"embed"
	"io/fs"
	"log"
	"math/rand"
	"net/http"
	"time"
)

// webFS stores all frontend files directly inside the Go binary.
// This means students can run one executable and still get both:
// 1) a backend API, and
// 2) a browser-based UI.
//
//go:embed web/*
var webFS embed.FS

func main() {
	// Seed random numbers once at program startup.
	// We use randomness for parameter initialization and probabilistic sampling.
	rand.Seed(time.Now().UnixNano())

	webRoot, err := fs.Sub(webFS, "web")
	if err != nil {
		log.Fatalf("failed to load web assets: %v", err)
	}

	server := NewServer()
	server.RegisterRoutes(http.DefaultServeMux, webRoot)

	log.Println("Server starting on :8080...")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatalf("server crashed: %v", err)
	}
}
