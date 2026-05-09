"""One-time compatibility probe: video metadata + face detection on source image and a sample video frame."""
import sys
import cv2
from PIL import Image

img_path = r'C:\Users\evija\faceswap\source\sreeni.jpg'
vid_path = r'C:\Users\evija\faceswap\songs\dekha-ek-khwab.mp4'

# --- video probe (OpenCV — same lens FaceFusion uses) ---
v = cv2.VideoCapture(vid_path)
ok = v.isOpened()
w = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = v.get(cv2.CAP_PROP_FPS)
n = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = int(v.get(cv2.CAP_PROP_FOURCC))
codec = bytes([(fourcc >> i * 8) & 0xff for i in range(4)]).decode('ascii', 'replace')
print('=== VIDEO ===')
print('opened:', ok)
print(f'resolution: {w}x{h}')
print(f'fps: {fps:.3f}')
print(f'frames: {n}')
print(f'duration: {n/fps:.1f}s' if fps else 'duration: unknown')
print(f'codec_fourcc: {codec}')
sample_t_s = (n / fps) / 2 if fps else 60
v.set(cv2.CAP_PROP_POS_FRAMES, int(sample_t_s * fps))
ok2, sample_frame = v.read()
print(f'mid_frame_read_ok: {ok2}')
v.release()

# --- image probe ---
im = Image.open(img_path)
print('\n=== SOURCE IMAGE ===')
print('format:', im.format, '| mode:', im.mode, '| size:', im.size)
src_bgr = cv2.imread(img_path)
print(f'cv2 shape: {src_bgr.shape}')

# --- face detection (insightface — same backbone DLC and FaceFusion use) ---
try:
    from insightface.app import FaceAnalysis
except Exception as e:
    print('\n[skip] insightface not installed in this env:', e)
    sys.exit(0)

print('\n=== FACE DETECTION ===')
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

src_faces = app.get(src_bgr)
print(f'faces in source image: {len(src_faces)}')
for i, f in enumerate(src_faces):
    print(f'  [{i}] bbox={f.bbox.astype(int).tolist()} score={f.det_score:.3f} '
          f'gender={"M" if f.sex=="M" else "F"} age={int(f.age)}')

if ok2:
    tgt_faces = app.get(sample_frame)
    print(f'faces in mid-video frame: {len(tgt_faces)}')
    for i, f in enumerate(tgt_faces):
        print(f'  [{i}] bbox={f.bbox.astype(int).tolist()} score={f.det_score:.3f} '
              f'gender={"M" if f.sex=="M" else "F"} age={int(f.age)}')

print('\n=== VERDICT ===')
ok_src = len(src_faces) >= 1
ok_tgt = ok2 and len(tgt_faces) >= 1
print(f'source has detectable face: {ok_src}')
print(f'target has detectable face (sample): {ok_tgt}')
print('compatible:', ok_src and ok_tgt)
