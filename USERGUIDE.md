# User Guide

A walkthrough of the web app for end users. **You don't need to know
Python.** If you can drag a file into a browser, you can use this.

> Looking for the technical / build-and-run side? See
> [README.md](README.md) and [CLAUDE.md](CLAUDE.md).

---

## Table of contents

1. [Before you start — what you need](#before-you-start--what-you-need)
2. [Step-by-step walkthrough](#step-by-step-walkthrough)
3. [What the screen shows you, in plain English](#what-the-screen-shows-you-in-plain-english)
4. [Tips for the best results](#tips-for-the-best-results)
5. [Duets — using two faces](#duets--using-two-faces)
6. [Frequently asked questions](#frequently-asked-questions)
7. [Common problems and fixes](#common-problems-and-fixes)
8. [Privacy, ethics, what NOT to do](#privacy-ethics-what-not-to-do)

---

## Before you start — what you need

1. **A computer the app is already installed on.** If yours isn't,
   ask whoever set it up, or follow [README.md](README.md).
2. **A face photo** of yourself. Phone selfie is fine. PNG or JPG.
   Front-facing, evenly lit, no sunglasses.
3. **A video** you want your face swapped into. MP4 is best. Up to
   ~4 GB.
4. **A web browser.** Chrome, Edge, Firefox, or Safari. Mobile
   browsers also work for **viewing** but you upload from a desktop.

The app runs at <http://localhost:8080/> on the machine where it's
installed. If you set it up, open that link.

---

## Step-by-step walkthrough

### 1. Open the app

Open <http://localhost:8080/> in your browser.

You'll see a single page with a glowing background, two areas to drop
your face image, and one area to drop the video.

### 2. Drag your face image into "Face #1"

Drag a photo of yourself onto the **Face #1 (required)** zone. You
should see:

- A small thumbnail of the photo appear inside the zone
- The filename and size below the thumbnail (e.g. `me.jpg · 2.3 MB`)
- The dashed border turn solid green
- The hint changes to "Looks good — ready to swap"

If you'd rather click than drag: just click anywhere in the zone, a
file picker opens.

### 3. (Optional) Drag a second face into "Face #2"

If your video is a duet — two leads of different gender, like a
classic Bollywood song — drop a second face image into **Face #2
(optional)**. The app will auto-figure out which face goes onto
which person in the video based on gender.

If your video is a solo or you don't care about the second person,
**leave Face #2 empty**. The app will only swap the matching-gender
person.

### 4. Drag the target video into "Target video"

The video can be any common format (MP4, MOV, MKV, WebM…) up to a few
GB. The thumbnail area will preview the first frame.

### 5. Click "Start live swap"

The button turns to "Uploading…" while your files transfer to the
server. On a local machine that's nearly instant.

### 6. Watch the progress

The page redirects to a viewer with five phases. Each lights up as it
runs:

| Phase | What's happening | Typical time |
|---|---|---|
| **load models** | First time only — pulls the swap models into VRAM. After the first run, this is instant | ~30 s once, then 0 s |
| **detect face** | Finds your face in the photo, reads gender + age | <1 s |
| **find reference** | Scans the video for the matching-gender face that recurs the most. That becomes the "lock" | 30–60 s |
| **stream** | The live HLS player appears here. The swap is happening in real-time | runs for the full song length, slightly longer |
| **finalise** | Builds the final downloadable MP4 | ~5 s |

### 7. Wait for the player to start

After 15 seconds of the **stream** phase, the player auto-plays —
**muted**. Browsers block sound on autoplay until you interact.

You'll see a small "🔊 Click to unmute" pill at the bottom-left of
the video. **Click it to hear the song.**

(Or click anywhere on the video — the page is also wired to start
playback on any click as a rescue.)

### 8. Watch the swap happen live

The video plays at its normal speed with the original audio. Your face
appears on the matching person whenever they're on screen. Frames
where neither lead is visible (intro shots, scenery, etc.) pass
through unchanged.

The progress bar below the player shows where in the song the swap
loop currently is — usually a few seconds ahead of what's playing in
the video. This is the buffer keeping playback smooth.

### 9. Download the finished MP4

When the **finalise** phase completes, a green "Your swap is ready"
card appears with a big **Download MP4 (with audio)** button. Click
it; you get a standard `.mp4` file that:

- Plays on every phone (iPhone, Android)
- Plays in VLC, QuickTime, Windows Media Player
- Can be uploaded to YouTube, Instagram, WhatsApp, etc.
- Is the same content the live HLS stream played, just re-packaged

The viewer page also stays usable as a VOD player — full scrub bar,
pause, replay — so you can re-watch without re-downloading.

---

## What the screen shows you, in plain English

When the job is running you'll see numbers like:

```
progress 3140 / 6664   fps 12.4   swaps 482
src1 F/40 (f3700, 22/28)   src2 M/44 (f300, 24/33)
```

Translation:

| Field | Meaning |
|---|---|
| **progress 3140 / 6664** | Frame 3140 out of 6664 has been processed (~47 %) |
| **fps 12.4** | The processor is working through 12.4 frames per second of source video |
| **swaps 482** | 482 frames so far had a face that the app swapped — frames with no detected face just pass through |
| **src1 F/40** | Source 1 was detected as Female, ~40 years old |
| **(f3700, 22/28)** | Source 1's "lock target" was found at frame 3700 of the video; 22 of 28 sample frames clustered onto that face — a very confident lock |
| **src2 M/44** | Source 2 was detected as Male, ~44 years old |

If you only uploaded one source, you'll just see `src1`.

---

## Tips for the best results

### For the source photo

- **Use a clear, front-facing photo** with the face filling at least
  ~25 % of the image. Selfies are great. Group photos cropped to one
  face are fine.
- **Even lighting**, no harsh shadows, no heavy filters.
- **Neutral expression** works best — open eyes, mouth slightly closed
  or a small smile. Big expressions can transfer to the swapped frames
  in odd ways.
- **No sunglasses, no hat brims covering the face, no masks.**
- **1024 × 1024 or larger.** The model only sees a 128 × 128 crop
  internally so don't worry about going huge — but bigger than 512 px
  on the face is the floor.

### For the target video

- **Higher resolution = better quality but slower.** 1080p source
  produces 1080p output but takes ~3× longer than 480p.
- **Sharp video**, not blurry — the face detector struggles with very
  blurry or low-contrast frames.
- **The leads should be visible for most of the song.** If they're
  only on screen for 20 % of the runtime, only 20 % of frames will
  contain a swap.
- **Fast cuts and dance shots are fine** — every frame is processed
  independently, so cuts don't break anything.

### For duets

- **Upload one face per gender.** The app uses gender to decide which
  source goes onto which lead.
- **If both leads are the same gender**, it'll work but the assignment
  is greedy: whichever cluster is bigger goes to source 1, the next
  biggest to source 2.
- **The order you upload them doesn't matter** — they're matched by
  gender, not slot.

---

## Duets — using two faces

Drop both faces (one per drop-zone) and start the swap. Behind the
scenes:

1. Each source image's face is detected, gender + age recorded
2. The video is scanned once; every face's embedding is captured
3. For each gender that's needed, the largest face cluster is found
4. Each source claims its gender's biggest cluster — and "claims"
   means similar candidates are marked used so a second same-gender
   source can take a different cluster
5. Per frame: every detected face is matched to the closest
   source-reference by face embedding similarity
6. Each face is swapped with its matched source's image

It "just works" for the common case (M + F duet). For unusual cases
(both leads same gender, three+ leads) the same logic applies — the
algorithm is greedy, it'll do its best.

---

## Frequently asked questions

### How long does it take?

Roughly **the song's length × 2–3 on a fast GPU**. A 4-minute song
takes 8–12 minutes wall-clock to fully render.

But you can **start watching the live stream after the first 15
seconds** of the song are processed (~30 s wait). You don't have to
wait for the whole render.

### Can I swap multiple videos with the same face?

Yes — refresh the page after each finishes, upload the same face,
pick a different video. The face models stay loaded (no reload time
on subsequent runs, just the model warmup once per server restart).

### Can I cancel a running swap?

Closing the browser tab doesn't cancel the server-side job. To
actually stop it, you'd need to restart the webapp from the
terminal. (We could add a Cancel button — file an issue.)

### Why does the video play **muted** at first?

Every modern browser refuses to play media with sound until you
interact with the page (so random tabs don't blast audio at you).
The app starts muted so playback at least begins; you click "🔊
Click to unmute" or anywhere on the video to enable audio.

### Why does the live stream sometimes pause briefly?

If the swap pipeline temporarily can't keep up (e.g. very dense
crowd shot in 1080p), the live edge moves slower than playback. The
player shows "Re-buffering 4 / 8 s…" and resumes when the buffer
recovers. This is normal — the saved download will be smooth.

### Why is the live preview slightly behind real-time?

Two reasons: (1) we pre-buffer 15 seconds before pressing play so
playback is smooth, and (2) the swap pipeline produces frames at
~10–15 fps wall-clock for a 25 fps source — so it falls a bit behind
over the run.

The **download** is always full speed and full quality.

### Can I use this in real-time on my webcam?

The web app is for files. If you want real-time, use the
`play-song.ps1` script in the repo, which launches Deep-Live-Cam's
GUI with a real-time webcam swap mode. See [README.md](README.md)
section on path B.

### Does the app upload my files to the cloud?

**No.** Everything runs on your local machine. The "server" is just
a Flask app on `localhost:8080`. No internet calls during processing
(model downloads happen once at install time).

### Where are my files stored?

Each upload creates a directory `webapp_jobs/<random-id>/` with:
- `source_0.jpg`, `source_1.jpg` — your uploaded face(s)
- `target.mp4` — your uploaded video
- `hls/playlist.m3u8` + `hls/seg_*.ts` — the live HLS stream
- `swapped.mp4` — the final downloadable MP4
- `ffmpeg.log` — for debugging if something goes wrong

Old job dirs are auto-deleted after 6 hours.

---

## Common problems and fixes

| Symptom | Cause | Fix |
|---|---|---|
| Page won't load (`localhost:8080` refuses) | Webapp isn't running | Start it: `conda run -n dlc python webapp.py` |
| Upload starts but nothing happens | Source image has no detectable face | Use a clearer, front-facing photo |
| Phase stuck on "find reference" forever | Video has no face of the matching gender | Upload a different source photo, OR a video that actually contains your gender |
| Live player shows progress but no video | Autoplay was blocked | Click anywhere on the video, or click the "🔊 Click to unmute" pill |
| Live player keeps stalling every few seconds | Pipeline is slower than playback (low-end GPU or 4K input) | Wait for the full render; the downloaded MP4 is smooth |
| Wrong person being swapped (e.g. female lead instead of male) | Source photo's gender mis-detected | Use a clearer photo where the face is bigger / better-lit |
| Downloaded MP4 won't play on my phone | Old buggy build (commits before `40ba7a1`) | Re-run the swap on the current version — the MP4 is now standard format |
| Audio plays but no video on the downloaded MP4 | Same as above — old fragmented MP4 | Re-run on current version |
| App is slow / GPU not being used | TRT or CUDA didn't load | See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |

If your symptom isn't here, see the deeper troubleshooting in
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md), or check the
ffmpeg log for that specific job at
`webapp_jobs/<job-id>/ffmpeg.log`.

---

## Privacy, ethics, what NOT to do

This tool is **for personal entertainment, education, and creative
projects with consent**.

**Do not** use it to:

- Make sexual deepfakes of anyone, ever.
- Impersonate real people in videos meant to deceive (fake news,
  defamation, fraud).
- Make videos of minors.
- Swap a real person's face onto another body without their explicit
  permission, especially if the result will be shared anywhere.

Many countries have specific laws about synthetic media of real
people; some require consent of all depicted persons before
distribution. Know yours.

If you're using this for a fun project (you and friends as Bollywood
stars, you in a movie scene as a gift, parody/satire under fair use,
etc.) — that's the intended use. Have fun.

---

## Where to go next

- **Want to know how it works under the hood?** [DESIGN.md](DESIGN.md)
  and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **Want to make it faster?** [docs/PERFORMANCE.md](docs/PERFORMANCE.md).
- **Want to extend or contribute?** [docs/HACKING.md](docs/HACKING.md)
  and [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).
- **Want to use this in OBS for a video call?** [OBS-setup.md](OBS-setup.md).
