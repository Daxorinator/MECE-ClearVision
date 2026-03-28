#!/usr/bin/env bash
# Download MediaPipe FaceMesh with attention ONNX (478 landmarks, includes iris)
# from PINTO0309's model zoo.  Places the file at src/models/.
#
# Usage: bash scripts/download_facemesh.sh
#
# Compatible with Git 2.17+ (Ubuntu 18.04 / JetPack 4.x).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../src/models"
mkdir -p "$MODELS_DIR"

OUT="$MODELS_DIR/face_landmark_with_attention.onnx"

if [ -f "$OUT" ]; then
    echo "Model already exists: $OUT"
    exit 0
fi

echo "Fetching FaceMesh download script from PINTO0309 model zoo..."

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

# Fetch just the download.sh for model #032 via raw GitHub URL (no git clone needed)
DOWNLOAD_SH_URL="https://raw.githubusercontent.com/PINTO0309/PINTO_model_zoo/main/032_face_landmark_with_attention/download.sh"

if ! curl -fsSL "$DOWNLOAD_SH_URL" -o "$TMP/download.sh"; then
    echo "ERROR: failed to fetch download.sh from PINTO0309 model zoo."
    echo "Check network connectivity or download manually:"
    echo "  https://github.com/PINTO0309/PINTO_model_zoo/tree/main/032_face_landmark_with_attention"
    echo "Place face_landmark_with_attention.onnx (192×192 input) in src/models/"
    exit 1
fi

echo "Running PINTO0309 download script..."
cd "$TMP"
bash download.sh

# Find the downloaded ONNX (prefer 192x192 variant, then any with_attention)
ONNX=$(find "$TMP" -name "*with_attention*192*.onnx" 2>/dev/null | head -1)
if [ -z "$ONNX" ]; then
    ONNX=$(find "$TMP" -name "*with_attention*.onnx" 2>/dev/null | head -1)
fi
if [ -z "$ONNX" ]; then
    echo "Available files in download directory:"
    find "$TMP" -name "*.onnx" 2>/dev/null | head -20
    echo
    echo "ERROR: could not find face_landmark_with_attention ONNX."
    echo "Download manually from:"
    echo "  https://github.com/PINTO0309/PINTO_model_zoo/tree/main/032_face_landmark_with_attention"
    echo "and place it at: $OUT"
    exit 1
fi

cp "$ONNX" "$OUT"
echo
echo "Saved: $OUT"
echo "Build view_synthesis and run — TRT engine is built on first launch (~2-5 min on Nano)."
echo "Cached engine: ${OUT}.trt"
