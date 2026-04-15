#!/bin/sh

# Create separate folders
MODEL_DIR="models"
VIDEO_DIR="videos"

mkdir -p "$MODEL_DIR"
mkdir -p "$VIDEO_DIR"

ARTIFACTS="
posenet_resnet50_uint8_float32_quant.tflite|$MODEL_DIR|https://github.com/NXP/nxp-demo-experience-assets/raw/lf-5.15.52_2.1.0/models/posenet_resnet50_uint8_float32_quant.tflite
key_point_labels.txt|$MODEL_DIR|https://github.com/NXP/nxp-demo-experience-assets/raw/lf-5.15.52_2.1.0/models/key_point_labels.txt
pose_example.mov|$VIDEO_DIR|https://github.com/NXP/nxp-demo-experience-assets/raw/lf-5.15.52_2.1.0/videos/pose_example.mov
"

echo "======================================"
echo " Downloading Artifacts"
echo " Models -> $MODEL_DIR"
echo " Videos -> $VIDEO_DIR"
echo "======================================"

echo "$ARTIFACTS" | while IFS="|" read -r NAME DEST URL; do
    [ -z "$NAME" ] && continue
    echo "Downloading: $NAME -> $DEST"

    if command -v curl >/dev/null 2>&1; then
        curl -L -o "$DEST/$NAME" "$URL"
    else
        wget -O "$DEST/$NAME" "$URL"
    fi

    if [ -f "$DEST/$NAME" ]; then
        echo "✅ $NAME downloaded successfully."
    else
        echo "❌ Failed to download $NAME."
    fi
done

echo "======================================"
echo " All downloads completed."
echo " Models saved in: $MODEL_DIR/"
echo " Videos saved in: $VIDEO_DIR/"
echo "======================================"
