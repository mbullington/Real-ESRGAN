#!/usr/bin/env bash
set -euo pipefail

IN_DIR="${1:-./in}"
shopt -s nullglob
for invid in "$IN_DIR"/*.{mp4,mkv,mov,avi,webm,flv}; do
    echo ">>> Upscaling: $invid"
    python inference_realesrgan_video.py -n OpenProteus -s 2 -i "$invid" -o "$IN_DIR"_out -t 0 --batch 3 --encoder hevc_vaapi --qp 22 --tile_pad 8 --queue_size 16
    echo ">>> Done: $invid"
done
