faceswap-mp — multiprocessing face-swap web app (installable)
=============================================================

Quick start (fresh Windows machine with NVIDIA GPU):

  1. Unzip this file anywhere — e.g. C:\faceswap-mp\
  2. Open PowerShell IN that folder.
     (Right-click in Explorer with Shift held → "Open PowerShell window here")
  3. If your PowerShell blocks scripts, run this once:
       Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force
  4. Run the installer:
       .\install.ps1
     This takes 10-15 minutes the first time. It installs Miniconda + ffmpeg
     via winget, creates a Python environment, downloads pip dependencies,
     and copies the bundled face models into ~/.insightface/models/.
  5. Once "INSTALL COMPLETE" appears, start the server:
       .\start.ps1
  6. Open http://localhost:8082/ in your browser. Drop in a face photo
     (or two: one male, one female) and a target video. Click "Start live
     swap" and watch the result stream while it's still rendering.


What's in this zip
------------------
  install.ps1                          → one-shot installer (run once)
  start.ps1                            → launches the web app (run every time)
  requirements-mp.txt                  → pinned pip deps used by install.ps1
  src/webapp_mp.py                     → the Flask web app + viewer HTML
  src/server/swap_worker.py            → per-process worker entry point
  src/server/__init__.py               → makes 'server' a Python package
  models/buffalo_l/*.onnx              → insightface face-analyser (290 MB)
  models/inswapper_128_fp16.onnx       → face-swap model (265 MB)
  tools/test-cuda-dlc.py               → CUDA-loaded-OK sanity check


Requirements on the target machine
----------------------------------
  - Windows 10 (build 19044+) or Windows 11
  - NVIDIA GPU with R535+ driver and >=8 GB VRAM (>=16 GB recommended)
  - winget (ships with Windows 10 21H2+ / Windows 11 — get it from the
    Microsoft Store if missing: search "App Installer")
  - Internet connection for the install (~2 GB of pip downloads)
  - ~5 GB free disk for the conda env + caches


Tuning knobs (env vars before running start.ps1)
------------------------------------------------
  FACESWAP_WORKERS     N parallel worker processes. Default 6. Drop to 4 on
                       a 16 GB VRAM card if you see out-of-memory.
  FACESWAP_DET_SIZE    Face-detector input edge. Default 480. Try 640 if
                       you see misses on small/far-away faces.
  FACESWAP_REF_THRESH  Cosine threshold for matching. Default 0.15. Lower
                       (0.10) for more frames swapped at risk of false
                       positives on extras.
  FACESWAP_PORT        Default 8082.

Example: spawn fewer workers and a stricter match threshold

   $env:FACESWAP_WORKERS = "4"
   $env:FACESWAP_REF_THRESH = "0.20"
   .\start.ps1


Troubleshooting
---------------
  "nvidia-smi not on PATH"
    Install the latest NVIDIA driver (Game Ready or Studio, either works):
    https://www.nvidia.com/Download/index.aspx — reboot — re-run install.ps1.

  "winget not available"
    Open the Microsoft Store, search "App Installer", install it. Reboot.
    Re-run install.ps1.

  "CUDA verification failed"
    The pip-installed cuDNN DLLs didn't load. Common causes:
      - Driver too old (need R535+); update the driver.
      - The 'dlc' or 'faceswap-mp' conda env was created with the wrong
        Python (must be 3.11); delete it with
          conda env remove -n faceswap-mp
        and re-run install.ps1.

  The first job takes 45 seconds before the fps counter moves
    Normal — workers are loading models in parallel. From the second frame
    onward you should see 20-30 fps at 1080p on a 4090 with N=6 workers.

  Browser shows "Buffering 0 / 4s" forever
    Should self-heal within 10s (a watchdog re-attaches HLS). If it doesn't,
    hit F5 to refresh the viewer page.


Source
------
  Full source + history: https://github.com/dlmastery/face-swap-streamer
  This installer is a minimal slice of that repo — webapp_mp.py only.
  For the C++ CLI, FastAPI/Next.js variant, or FaceFusion path, clone the
  repo and follow CLAUDE.md.
