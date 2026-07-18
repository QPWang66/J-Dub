#!/bin/sh
# Fetch the model weights jdub-cv needs (kept out of git).
# court_kp.pt from this repo's GitHub release; WASB ball weights from the
# release too (MIT, redistributed) with the authors' Google Drive as fallback.
set -e
cd "$(dirname "$0")"
mkdir -p weights

REL="https://github.com/QPWang66/J-Dub/releases/download/cv-weights-v1"

[ -f weights/court_kp.pt ] || curl -fL "$REL/court_kp.pt" -o weights/court_kp.pt

if [ ! -f weights/wasb_basketball_best.pth.tar ]; then
    curl -fL "$REL/wasb_basketball_best.pth.tar" -o weights/wasb_basketball_best.pth.tar \
    || uvx gdown 1nfECuSyJvPUmz3njZCdFERSQQbERt8FU -O weights/wasb_basketball_best.pth.tar
fi

echo "weights ready:"
ls -lh weights/
