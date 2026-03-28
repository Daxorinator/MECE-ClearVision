#!/usr/bin/env bash
# Download MediaPipe FaceMesh with attention ONNX (478 landmarks, includes iris)
# from PINTO0309's model zoo.  Places the file at src/models/.
#
# Usage: bash scripts/download_facemesh.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../src/models"
mkdir -p "$MODELS_DIR"

OUT="$MODELS_DIR/face_landmark_with_attention.onnx"

if [ -f "$OUT" ]; then
    echo "Model already exists: $OUT"
    exit 0
fi

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

echo "Downloading 282_face_landmark_with_attention resources..."
curl "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/282_face_landmark_with_attention/resources.tar.gz" \
    -o "$TMP/resources.tar.gz"
tar -zxvf "$TMP/resources.tar.gz" -C "$TMP"
rm "$TMP/resources.tar.gz"

# Find the ONNX (prefer 192x192 float32, then any .onnx in the archive)
ONNX=$(find "$TMP" -name "*192*float32*.onnx" 2>/dev/null | head -1)
if [ -z "$ONNX" ]; then
    ONNX=$(find "$TMP" -name "*.onnx" 2>/dev/null | head -1)
fi
if [ -z "$ONNX" ]; then
    echo "Available files:"
    find "$TMP" -type f | head -20
    echo
    echo "ERROR: no .onnx found in archive."
    echo "Download manually from:"
    echo "  https://github.com/PINTO0309/PINTO_model_zoo/tree/main/282_face_landmark_with_attention"
    echo "and place it at: $OUT"
    exit 1
fi

cp "$ONNX" "$OUT"
echo
echo "Saved: $OUT"
echo "Build view_synthesis and run — TRT engine is built on first launch (~2-5 min on Nano)."
echo "Cached engine: ${OUT}.trt"
