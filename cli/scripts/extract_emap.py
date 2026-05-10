"""Extract the inswapper emap (last graph initializer) to a raw float32 binary
that the C++ Inswapper can mmap at startup.

Run once after setup:
    conda run -n dlc python cli/scripts/extract_emap.py
"""
import sys
from pathlib import Path

import onnx
import numpy as np
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]  # cli/
MODEL = ROOT / "models" / "inswapper_128_fp16.onnx"
OUT = ROOT / "models" / "inswapper_emap.bin"

m = onnx.load(str(MODEL))
print(f"initializers: {len(m.graph.initializer)}")
last = m.graph.initializer[-1]
print(f"last name: {last.name}  dims: {list(last.dims)}  type: {last.data_type}")
arr = numpy_helper.to_array(last).astype(np.float32)
print(f"shape: {arr.shape}  min/max: {arr.min():.4f} / {arr.max():.4f}")

if arr.shape != (512, 512):
    sys.exit(f"unexpected emap shape {arr.shape}; expected (512, 512)")

arr.tofile(str(OUT))
print(f"wrote {arr.size * 4} bytes to {OUT}")
