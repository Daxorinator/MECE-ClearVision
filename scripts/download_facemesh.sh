#!/usr/bin/env bash
# Download MediaPipe FaceMesh with attention ONNX (478 landmarks, includes iris)
# from PINTO0309's model zoo.  Places the file at src/models/.
#
# Usage: bash scripts/download_facemesh.sh
#
# PINTO0309 model zoo #032: face_landmark_with_attention
# https://github.com/PINTO0309/PINTO_model_zoo/tree/main/032_face_landmark_with_attention

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../src/models"
mkdir -p "$MODELS_DIR"

OUT="$MODELS_DIR/face_landmark_with_attention.onnx"

if [ -f "$OUT" ]; then
    echo "Model already exists: $OUT"
    exit 0
fi

echo "Downloading FaceMesh with attention ONNX from PINTO0309 model zoo..."

# PINTO0309 distributes models via a download.sh that pulls from their releases.
# Clone just the relevant subdirectory using sparse-checkout (no LFS needed for ONNX).
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

git clone --filter=blob:none --sparse \
    https://github.com/PINTO0309/PINTO_model_zoo.git \
    "$TMP/pinto_model_zoo" 2>&1 | tail -5

cd "$TMP/pinto_model_zoo"
git sparse-checkout set 032_face_landmark_with_attention

# Run PINTO0309's own download script to fetch the actual ONNX files
cd 032_face_landmark_with_attention
bash download.sh 2>&1 | tail -20

# Copy the 192x192 ONNX to our models directory
ONNX=$(find . -name "*with_attention*192*.onnx" | head -1)
if [ -z "$ONNX" ]; then
    ONNX=$(find . -name "*with_attention*.onnx" | head -1)
fi
if [ -z "$ONNX" ]; then
    echo "ERROR: could not find face_landmark_with_attention ONNX in download output"
    echo "Files present:"
    find . -name "*.onnx" | head -20
    exit 1
fi

cp "$ONNX" "$OUT"
echo "Saved to: $OUT"
echo
echo "Build and run view_synthesis — TRT engine will be built on first launch (~2-5 min)."
echo "The compiled engine is cached at: ${OUT}.trt"
