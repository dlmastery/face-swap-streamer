# multi-face-swapper-streaming — design

Draft design for a new repo `multi-face-swapper-streaming` that builds on
`webapp_mp.py` with a face-grid UX where the user assigns swap photos
per detected face after the video has been scanned.

Status: design phase. Sections 1–2 validated. Sections 3+ pending.

## Goal

Same upload-and-stream feel as `webapp_mp.py`, but with the user in the
loop for matching: after upload the page shows up to 12 detected faces
ranked by how often they appear in the video, and the user assigns a
swap photo to each face they care about. Skipped faces pass through
unswapped. Then the existing multiprocess HLS streaming pipeline kicks
in unchanged.

## Repo layout

A NEW repo, not a branch of `face-swap-streamer`. Codebase descends
from this one (especially `webapp_mp.py` + `server/swap_worker.py`)
but the UX flow and APIs diverge enough that keeping them separate
is cleaner than feature-flagging.

## UX — single-page wizard (validated)

The page has 5 visible states, each replacing the previous in-place
(no navigation):

1. **`upload`** — drop-zone for the mp4 + "Scan faces" button. No source
   photos yet.
2. **`scanning`** — spinner + progress text: "Reading frames…
   (37 / 120)", "Clustering faces…", "Generating thumbnails…". Backend
   does the existing 120-frame sample + identity clustering pass we
   already have, just exposed as a separate phase. 10–15 s on a 4090
   with warm models.
3. **`assign`** — face grid appears. Cards ranked by `member_count`
   (most visible first):
   - 256×256 thumbnail of the cluster's representative frame
   - Caption: "Face #1 · appears in 47 frames"
   - Drag-drop zone INSIDE the card for the swap photo, OR a "Skip" toggle
   - Assigned cards turn green, skipped cards grey out
   - "Start swap" button at the bottom; disabled until ≥1 card has a photo
   - Cap at 12 faces (rest fall through to passthrough)
4. **`streaming`** — the existing HLS viewer replaces the grid. Same
   player + progress + Pause/Resume/Stop buttons from `webapp_mp.py`.
5. **`done`** — same as today: download MP4 link + replay.

Skipped faces → passthrough. The current matching code already does
this: target embedding with no source above ref_thresh → frame
forwarded unchanged.

## API — scan / swap split (validated)

Three new endpoints replace the current single `/start`:

```
POST /scan
  body: multipart with `target` (mp4 file)
  action: save mp4 to a scan dir, kick off a background scan worker:
    - opens video, samples up to 120 frames evenly
    - face_analyser.get() on each → embeddings + bboxes
    - identity-cluster (existing code from webapp_mp.py)
    - top-12 clusters by size
    - for each: pick highest-det-score member as rep frame, crop the
      face bbox + 30% margin, resize to 256×256, save as
      scans/<scan_id>/face_<N>.jpg
  returns: {scan_id, faces: [{id, frame_count, thumb_url}, ...]} once done
  poll for partial: GET /scan/<scan_id> returns same shape;
                    phase="scanning" until done.

POST /scan/<scan_id>/swap
  body: multipart with `face_<N>` files (one per face the user assigned).
        Missing face_<N> keys = passthrough (no swap on that cluster).
  action: build SourceSpec list from uploaded photos + map each to its
          cluster's ref_emb / ref_members (already computed by scan).
          Spawn workers (existing webapp_mp.py logic).
  returns: 302 redirect to /job/<job_id>

GET /job/<job_id>/status     same as today
GET /job/<job_id>/hls/...    same as today
GET /job/<job_id>/download   same as today
POST /job/<job_id>/pause     same as today
POST /job/<job_id>/resume    same as today
POST /job/<job_id>/stop      same as today
GET  /scans/<scan_id>/face_<N>.jpg   serves thumbnails (read-only)
```

Why split scan and swap:
- Scan output (cluster centroids + members + thumbnails) is cached on
  disk under `scans/<scan_id>/`. If the user clicks back or refreshes
  mid-assign, no re-scan needed.
- Workers don't need to spawn until step 3 — saves the 45 s warmup if
  the user changes their mind during assignment.
- Same `scan_id` could be re-used to launch multiple swap jobs against
  the same video with different face assignments — handy for iterating
  on which face to keep vs swap.

## Pending sections (TODO before implementation)

- **Section 3**: Thumbnail generation + face-card JSON payload shape.
- **Section 4**: Multi-source swap pipeline reuse. Verify the existing
  matching code (NN-over-members across N sources) scales to 12
  sources without re-design.
- **Section 5**: Edge cases — fewer than 12 clusters, user uploads zero
  source photos, target video has zero detected faces, etc.
- **Section 6**: Repo bootstrap — README, CLAUDE.md, install scripts,
  CI smoke test, initial commit structure.
- **Section 7**: Which of `face-swap-streamer`'s features migrate
  (Pause/Resume/Stop, HLS streaming, NVENC, the watchdog) vs which are
  dropped (FastAPI/Next.js variant, C++ CLI, batch upload — likely all
  out of scope for v1).
