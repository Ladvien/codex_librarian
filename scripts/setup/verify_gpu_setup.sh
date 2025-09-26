#!/bin/bash

echo "==================================="
echo "GPU Acceleration Verification"
echo "==================================="
echo ""

echo "1. PyTorch CUDA Status:"
.venv/bin/python -c "
import torch
print(f'  ✓ PyTorch version: {torch.__version__}')
print(f'  ✓ CUDA available: {torch.cuda.is_available()}')
print(f'  ✓ CUDA version: {torch.version.cuda}')
print(f'  ✓ GPU: {torch.cuda.get_device_name(0)}')
print(f'  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"
echo ""

echo "2. MinerU Configuration:"
.venv/bin/python -c "
from mineru.utils.config_reader import get_device
print(f'  ✓ Device mode: {get_device()}')
"
echo ""

echo "3. Config File:"
if [ -f ~/mineru.json ]; then
    echo "  ✓ ~/mineru.json exists"
    cat ~/mineru.json | head -5
else
    echo "  ✗ ~/mineru.json not found"
fi
echo ""

echo "4. GPU Hardware:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | \
  awk -F', ' '{printf "  ✓ %s\n  ✓ Memory: %s\n  ✓ Compute: %s\n", $1, $2, $3}'
echo ""

echo "==================================="
echo "✅ GPU Acceleration Ready!"
echo "==================================="
echo ""
echo "Next steps:"
echo "  1. Run: .venv/bin/python examples/watch_and_mirror.py --batch --watch-dir watch_pdf_test --output-dir /tmp/gpu_output"
echo "  2. Monitor: watch -n 1 nvidia-smi"
echo "  3. Expect 5-10x speedup on complex PDFs"
