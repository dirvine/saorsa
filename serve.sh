#!/bin/bash

# Simple script to serve the documentation website locally

PORT=${1:-8080}
cd "$(dirname "$0")/docs"

echo "Starting server on http://localhost:$PORT"
echo "Press Ctrl+C to stop"

python3 -m http.server $PORT