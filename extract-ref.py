"""Extract a frame from the target video that contains a clear male face,
to use as the --reference for stream-swap."""
import os, sys, glob

_dll_cookies = []
if sys.platform == "win32":
    _sp = os.path.join(sys.prefix, "Lib", "site-packages")
    for _sub in ("cudnn", "cublas", "cuda_runtime", "curand", "cufft", "cuda_nvrtc", "nvjitlink"):
        _bin = os.path.join(_sp, "nvidia", _sub, "bin")
        if os.path.isdir(_bin):
            try: _dll_cookies.append(os.add_dll_directory(_bin))
            except OSError: pass

import cv2
import argparse
from insightface.app import FaceAnalysis

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--out",   required=True)
ap.add_argument("--gender", default="M", choices=["M", "F"])
ap.add_argument("--min-bbox", type=int, default=80, help="min face width in pixels")
ap.add_argument("--scan-step-sec", type=float, default=2.0)
args = ap.parse_args()

fa = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
fa.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step = int(fps * args.scan_step_sec)

best = None
i = 0
while i < total:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read()
    if not ok:
        break
    faces = fa.get(frame)
    males = [f for f in faces if f.sex == args.gender]
    if males:
        # prefer larger, higher-confidence faces, with some age preference (skip < 25 → kids)
        males.sort(key=lambda f: ((f.bbox[2]-f.bbox[0]) * f.det_score), reverse=True)
        cand = males[0]
        w_face = cand.bbox[2] - cand.bbox[0]
        if w_face >= args.min_bbox and (best is None or w_face > best[0]):
            best = (w_face, cand, frame.copy(), i)
            print(f"[extract-ref] frame {i} t={i/fps:.1f}s "
                  f"face_w={int(w_face)} score={cand.det_score:.2f} age={int(cand.age)}", flush=True)
    i += step

cap.release()

if best is None:
    sys.exit(f"no {args.gender} face >={args.min_bbox}px found in video")

w_face, face, frame, idx = best
cv2.imwrite(args.out, frame)
print(f"[extract-ref] saved {args.out} (frame {idx}, {args.gender} face {int(w_face)}px)")
