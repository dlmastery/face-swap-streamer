"""Stream-swap: read MP4 frames, swap face on-the-fly, pipe live to ffplay.

Usage:
    python stream-swap.py --source <face.jpg> --target <video.mp4>

Pipeline: cv2 read -> insightface detect -> inswapper -> ffplay (subprocess pipe).
You see the swap happening as it processes — no fully-rendered MP4 first.
"""
from __future__ import annotations
import argparse
import os
import sys
import glob
import subprocess
import time

# --- Win Py 3.8+ secure DLL search: register CUDA dirs before importing onnxruntime ---
_dll_cookies = []
if sys.platform == "win32":
    _sp = os.path.join(sys.prefix, "Lib", "site-packages")
    for _sub in ("cudnn", "cublas", "cuda_runtime", "curand", "cufft",
                 "cuda_nvrtc", "nvjitlink"):
        _bin = os.path.join(_sp, "nvidia", _sub, "bin")
        if os.path.isdir(_bin):
            try:
                _dll_cookies.append(os.add_dll_directory(_bin))
            except OSError:
                pass
            os.environ["PATH"] = _bin + os.pathsep + os.environ["PATH"]

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# ---- MJPEG web streaming ---------------------------------------------------

# Latest JPEG bytes per active client. Single producer (main loop), many consumers.
_latest_jpeg: bytes | None = None
_jpeg_lock = threading.Lock()
_jpeg_event = threading.Event()


def _make_html(width: int, height: int, title: str) -> bytes:
    return f"""<!doctype html><html><head>
<meta charset='utf-8'><title>stream-swap: {title}</title>
<style>
  html,body {{ margin:0; padding:0; background:#111; color:#ddd;
    font-family: ui-sans-serif, system-ui, sans-serif; }}
  .wrap {{ display:flex; flex-direction:column; align-items:center;
    justify-content:center; min-height:100vh; gap:.5rem; padding:1rem; }}
  img {{ max-width:100%; max-height:90vh; box-shadow:0 0 30px #000;
    border-radius:6px; background:#000; }}
  small {{ opacity:.6 }}
</style></head><body><div class='wrap'>
<img src='/mjpeg' alt='live face-swap stream'>
<small>{title} &middot; {width}&times;{height}</small>
</div></body></html>""".encode("utf-8")


class _Handler(BaseHTTPRequestHandler):
    server_version = "stream-swap/1.0"
    width = height = 0
    title = ""

    def log_message(self, fmt, *args):  # silence default access logs
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = _make_html(self.width, self.height, self.title)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/mjpeg":
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last_sent = None
            try:
                while True:
                    _jpeg_event.wait(timeout=10)
                    with _jpeg_lock:
                        jpg = _latest_jpeg
                        _jpeg_event.clear()
                    if jpg is None or jpg is last_sent:
                        continue
                    self.wfile.write(b"--frame\r\n"
                                     b"Content-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                                     + jpg + b"\r\n")
                    last_sent = jpg
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except Exception as e:
                print(f"[stream-swap web] client error: {e}", flush=True)
            return
        self.send_error(404)


def start_web_server(port: int, width: int, height: int, title: str) -> ThreadingHTTPServer:
    _Handler.width = width
    _Handler.height = height
    _Handler.title = title
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def push_jpeg(frame_bgr: np.ndarray, quality: int = 80) -> None:
    global _latest_jpeg
    ok, buf = cv2.imencode(".jpg", frame_bgr,
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return
    with _jpeg_lock:
        _latest_jpeg = buf.tobytes()
    _jpeg_event.set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="source face image")
    ap.add_argument("--target", required=True, help="target MP4")
    ap.add_argument("--swapper", default=r"C:\Users\evija\faceswap\deep-live-cam\models\inswapper_128_fp16.onnx")
    ap.add_argument("--many", action="store_true", help="swap every face, not just first")
    ap.add_argument("--no-audio", action="store_true", help="skip audio (faster)")
    ap.add_argument("--scale", type=float, default=1.0, help="display scale (0.5 = half)")
    ap.add_argument("--save", help="optional: also save swapped MP4 to this path")
    ap.add_argument("--gender", choices=["M", "F", "any"], default="any",
                    help="filter detections to this gender (used during auto-reference, then drops to 'any' "
                         "since embedding match is more reliable)")
    ap.add_argument("--reference", help="path to a reference image of the target face to lock onto. "
                                        "If omitted, auto-extracted from the target video using --gender.")
    ap.add_argument("--reference-thresh", type=float, default=0.22,
                    help="min cosine similarity to reference embedding (0.22 = forgiving, 0.30 = strict)")
    ap.add_argument("--det-size", type=int, default=640,
                    help="face detector input size; 640 is the model's native — non-default values "
                         "can disable detection on portrait/non-square sources")
    ap.add_argument("--det-thresh", type=float, default=0.40,
                    help="face detector score threshold (0.5 = strict default, 0.4 = catches more)")
    ap.add_argument("--auto-ref-scan-sec", type=float, default=2.0,
                    help="seconds between sample frames when auto-extracting reference")
    ap.add_argument("--web", action="store_true",
                    help="stream to a browser via HTTP MJPEG instead of ffplay (open http://localhost:<port>/)")
    ap.add_argument("--web-port", type=int, default=8080)
    ap.add_argument("--web-quality", type=int, default=80,
                    help="JPEG quality for the MJPEG stream (0-100)")
    args = ap.parse_args()

    if not os.path.isfile(args.source):
        sys.exit(f"source not found: {args.source}")
    if not os.path.isfile(args.target):
        sys.exit(f"target not found: {args.target}")
    if not os.path.isfile(args.swapper):
        sys.exit(f"swapper model not found: {args.swapper}")

    # face analyser — try CUDA first, fall back to CPU
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print(f"[stream-swap] init face analyser ({providers[0]}, det_size={args.det_size}, det_thresh={args.det_thresh})...", flush=True)
    fa = FaceAnalysis(name="buffalo_l", providers=providers)
    fa.prepare(ctx_id=0, det_size=(args.det_size, args.det_size), det_thresh=args.det_thresh)

    # inswapper
    print(f"[stream-swap] load inswapper from {os.path.basename(args.swapper)}...", flush=True)
    swapper = insightface.model_zoo.get_model(args.swapper, providers=providers)

    # source face
    src_bgr = cv2.imread(args.source)
    if src_bgr is None:
        sys.exit(f"cannot read source: {args.source}")
    src_faces = fa.get(src_bgr)
    if not src_faces:
        sys.exit("no face detected in source image")
    src_face = max(src_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    print(f"[stream-swap] source face: {src_face.sex}/{int(src_face.age)} "
          f"bbox={src_face.bbox.astype(int).tolist()}", flush=True)

    # Open video early so we can auto-extract a reference if needed
    cap = cv2.VideoCapture(args.target)
    if not cap.isOpened():
        sys.exit(f"cannot open target: {args.target}")
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Reference embedding — explicit file, or auto-scan video for clearest matching face
    ref_emb = None
    if args.reference:
        ref_bgr = cv2.imread(args.reference)
        if ref_bgr is None:
            sys.exit(f"cannot read reference: {args.reference}")
        ref_faces = fa.get(ref_bgr)
        if args.gender != "any":
            ref_faces = [f for f in ref_faces if f.sex == args.gender]
        if not ref_faces:
            sys.exit(f"no {args.gender} face in reference image")
        ref_face = max(ref_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        ref_emb = ref_face.normed_embedding
        print(f"[stream-swap] reference: {ref_face.sex}/{int(ref_face.age)} "
              f"from {os.path.basename(args.reference)} thresh={args.reference_thresh}", flush=True)
    elif args.gender != "any":
        # auto-scan video for the best representative of the requested gender
        print(f"[stream-swap] auto-extracting {args.gender} reference from video...", flush=True)
        step = max(1, int(fps * args.auto_ref_scan_sec))
        candidates = []  # (score, embedding, frame_idx, bbox)
        i = 0
        while i < total and len(candidates) < 60:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok:
                break
            faces = [f for f in fa.get(fr) if f.sex == args.gender]
            if faces:
                best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * f.det_score)
                w_face = best.bbox[2] - best.bbox[0]
                if w_face >= 50:  # ignore tiny detections
                    candidates.append((float(w_face * best.det_score),
                                       best.normed_embedding, i, best.bbox))
            i += step
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not candidates:
            sys.exit(f"could not auto-extract {args.gender} reference from video")
        # cluster by embedding — pick the medoid of the largest cluster (most-recurring face)
        embs = np.stack([c[1] for c in candidates])
        # similarity matrix
        sim = embs @ embs.T
        # for each candidate, count how many others are similar (sim > 0.30)
        votes = (sim > 0.30).sum(axis=1)
        winner = int(np.argmax(votes * np.array([c[0] for c in candidates])))
        ref_emb = candidates[winner][1]
        ref_frame = candidates[winner][2]
        ref_score = candidates[winner][0]
        print(f"[stream-swap] auto-reference: frame {ref_frame} ({ref_frame/fps:.1f}s), "
              f"votes={int(votes[winner])}/{len(candidates)}, score={ref_score:.0f}, "
              f"thresh={args.reference_thresh}", flush=True)

    # video already opened above for auto-reference; just compute output size
    out_w, out_h = int(in_w * args.scale), int(in_h * args.scale)
    print(f"[stream-swap] video: {in_w}x{in_h} @ {fps:.2f}fps, {total} frames "
          f"(streaming at {out_w}x{out_h})", flush=True)

    ffplay = None
    web_server = None
    if args.web:
        web_server = start_web_server(args.web_port, out_w, out_h, os.path.basename(args.target))
        print(f"[stream-swap] web server up — open http://localhost:{args.web_port}/ in your browser", flush=True)
    else:
        # Find a working ffplay (Anaconda's is missing SDL2 deps on this machine)
        candidates = [
            r"C:\Users\evija\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffplay.exe",
            "ffplay",
        ]
        ffplay_exe = next((c for c in candidates if c == "ffplay" or os.path.isfile(c)), None)
        if ffplay_exe is None:
            sys.exit("ffplay not found")
        print(f"[stream-swap] using ffplay: {ffplay_exe}", flush=True)
        ffplay_cmd = [
            ffplay_exe, "-hide_banner", "-loglevel", "warning",
            "-window_title", f"stream-swap: {os.path.basename(args.target)}",
            "-x", str(out_w), "-y", str(out_h),
            "-fflags", "nobuffer", "-flags", "low_delay",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{out_w}x{out_h}", "-framerate", str(fps),
            "-autoexit", "-i", "-",
        ]
        print(f"[stream-swap] spawning ffplay (video only stream)...", flush=True)
        ffplay = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE)

    # optional file writer (mp4) for keeping a copy of what's streamed
    writer = None
    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (out_w, out_h))

    t0 = time.time()
    n = 0
    swapped_count = 0
    last_print = t0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1

            # detect target faces
            tgt_faces = fa.get(frame)

            # Reference embedding match (more reliable than gender) — pick the face
            # whose embedding is closest to the locked reference. This works even when
            # the gender prediction flips on profile/low-res faces.
            picked = None
            best_sim = -1.0
            if ref_emb is not None and tgt_faces:
                for f in tgt_faces:
                    sim = float(np.dot(f.normed_embedding, ref_emb))
                    if sim > best_sim:
                        best_sim = sim
                        picked = f
                if best_sim < args.reference_thresh:
                    picked = None
            elif tgt_faces:
                # no reference — fall back to gender filter then largest
                pool = tgt_faces
                if args.gender != "any":
                    pool = [f for f in pool if f.sex == args.gender]
                if pool:
                    picked = max(pool, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

            if picked is not None:
                frame = swapper.get(frame, picked, src_face, paste_back=True)
                swapped_count += 1

            if args.scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            if ffplay is not None:
                try:
                    ffplay.stdin.write(frame.tobytes())
                except (BrokenPipeError, OSError):
                    print("[stream-swap] ffplay closed — stopping", flush=True)
                    break
            if web_server is not None:
                push_jpeg(frame, args.web_quality)
            if writer is not None:
                writer.write(frame)

            now = time.time()
            if now - last_print > 1.0:
                elapsed = now - t0
                cur_fps = n / elapsed
                eta = (total - n) / cur_fps if cur_fps > 0 else 0
                print(f"[stream-swap] frame {n}/{total} ({100*n/total:.1f}%) "
                      f"swap={swapped_count} | {cur_fps:.1f}fps | eta {eta:.0f}s",
                      flush=True)
                last_print = now
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if ffplay is not None:
            try:
                ffplay.stdin.close()
            except Exception:
                pass
            try:
                ffplay.wait(timeout=5)
            except Exception:
                pass
        if web_server is not None:
            print("[stream-swap] press Ctrl+C in this terminal to stop the web server "
                  "(it stays up so you can rewind with the saved file)", flush=True)

    elapsed = time.time() - t0
    print(f"[stream-swap] done: {n} frames in {elapsed:.1f}s "
          f"({n/elapsed:.1f}fps), {swapped_count} swapped", flush=True)

    # Mux original audio into the saved video (cv2.VideoWriter is video-only)
    if args.save and os.path.isfile(args.save):
        muxed = os.path.splitext(args.save)[0] + "_audio.mp4"
        ffmpeg_exe = next((p for p in [
            r"C:\Users\evija\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe",
            "ffmpeg",
        ] if p == "ffmpeg" or os.path.isfile(p)), None)
        if ffmpeg_exe:
            print(f"[stream-swap] muxing audio from source -> {muxed}", flush=True)
            mux_cmd = [
                ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
                "-i", args.save,                # video (no audio)
                "-i", args.target,              # original (for audio)
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "copy",                 # keep encoded video as-is
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                muxed,
            ]
            rc = subprocess.call(mux_cmd)
            if rc == 0:
                print(f"[stream-swap] muxed file ready: {muxed}", flush=True)
            else:
                print(f"[stream-swap] ffmpeg mux failed (rc={rc})", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
