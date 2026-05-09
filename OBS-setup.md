# Path C — Stream a swapped MP4 as a virtual webcam (OBS)

Use this once you've produced a swapped MP4 with `swap-song.ps1` (Path A). It loops the file and exposes it as a fake webcam to Discord, Zoom, Google Meet, Teams, OBS Live, etc.

## 1. Install OBS Studio (one-time)

Download: <https://obsproject.com/download> → Windows → run installer. Free, no signup.

## 2. Add your swapped video as a Media Source

1. Open OBS.
2. **Sources** panel → `+` → **Media Source** → name it `bollywood-swap`.
3. **Local File** → browse to `C:\Users\evija\faceswap\out\<song>_swapped.mp4`.
4. ✅ **Loop**
5. ✅ **Restart playback when source becomes active**
6. ✅ **Use hardware decoding when available**
7. OK.

To switch songs: select the source → properties → change file. Or make multiple Media Sources, one per song, and toggle their visibility (eye icon).

## 3. Make it look good in the Preview

1. **Right-click the source** → **Transform** → **Fit to Screen** (`Ctrl+F`) so it fills the canvas.
2. If your canvas isn't 1920x1080: Settings → Video → set Base + Output Resolution to match the swapped MP4 (likely 1920x1080).
3. Optional: add **Filters** → **Color Correction** to match your room lighting if you'll be on a real call.

## 4. Enable Virtual Camera

1. Bottom-right of OBS: **Start Virtual Camera**.
2. Verify it's running — the button turns red and says **Stop Virtual Camera**.

## 5. Use it everywhere

In any app's video device picker, choose **OBS Virtual Camera**:

- **Discord**: User Settings → Voice & Video → Camera → OBS Virtual Camera
- **Zoom**: Settings → Video → Camera → OBS Virtual Camera
- **Google Meet**: in-call camera switcher (top right) → OBS Virtual Camera
- **Microsoft Teams**: Settings → Devices → Camera → OBS Virtual Camera
- **Streamlabs / OBS Live / TikTok Live Studio**: same — pick OBS Virtual Camera as the input

## 6. Tips

- **Audio**: Media Source includes the song's audio — OBS routes it to "Desktop Audio" by default. To send it to the call's microphone, add a VB-Audio CABLE virtual mic and route OBS Audio Monitoring → CABLE Input. Most call apps don't expose OBS audio directly; the audio question is independent of the face swap.
- **Performance**: Since the MP4 is pre-rendered (Path A did the GPU work already), OBS playback is trivial — under 5% CPU.
- **Lower latency loop**: Right-click the source → **Properties** → uncheck **Show nothing when playback ends** → set Speed = 100 → check **Use hardware decoding** → ✅ Loop. The seam between loops is ~50ms imperceptible.
- **Multi-song queue**: create a Scene per song, hotkey-switch with `Settings → Hotkeys`.
- **Cropping the watermark / branding**: drag red handles in the preview to crop, or use **Filters → Crop/Pad**.

## 7. Real-time mode (Path B) instead

If you want the swap done **live during playback** rather than from a pre-rendered file (slightly different latency / quality tradeoff), use `play-song.ps1` (Deep-Live-Cam Real-time Playback) and capture *that* preview window in OBS via a **Window Capture** source instead of a Media Source. Higher GPU load during the call.
