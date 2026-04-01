#!/bin/bash
# Download sample soccer video for testing
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"

mkdir -p "$DATA_DIR"

URL="https://drive.google.com/uc?export=download&id=1d-7gXcH1Zy-DNi2yZ56zfgiqayz3cuM2"
OUTPUT="$DATA_DIR/sample.mp4"

if [ -f "$OUTPUT" ]; then
    echo "Sample video already exists at $OUTPUT"
    exit 0
fi

echo "Downloading sample soccer video..."
curl -L -o "$OUTPUT" "$URL"
echo "Done! Saved to $OUTPUT"
