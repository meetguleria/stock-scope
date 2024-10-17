package main

import (
  "fmt"
  "net/http"
  "time"
)

func greet(w http.ResponseWriter, r *http.Request) {
  fmt.Fprintf(w, "Hello World! %s", time.Now())
}


func main() {
  http.HandleFunc("/", greet)
  fmt.Println("Server is running on http://localhost:8080")
  http.ListenAndServe(":8080", nil)
}