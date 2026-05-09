# Contributing

Want to push a change? Read [HACKING.md](HACKING.md) first if you're
new — it covers the editing workflow and code map. This doc is the
contribution-process side: how to commit, what to test, what we will
and won't accept.

---

## Before you open a PR

1. **Did you run the smoke test?** ([HACKING.md § How to test a change](HACKING.md#how-to-test-a-change))
2. **Did the live HLS play with audio after click-to-unmute?**
3. **Does the downloaded MP4 play in VLC AND on a phone?**
4. **Did you re-read the diff with fresh eyes?** Especially anything
   touching `_run_job`'s `finally:` block — that's where deadlocks
   hide.

If yes to all four, open the PR.

---

## What we'll accept

| Change | Wanted? | Notes |
|---|---|---|
| Bug fixes | ✅ Yes | Especially around DLL discovery, HLS, autoplay |
| Documented speedups | ✅ Yes | Include before/after fps numbers in the commit message |
| New optional features behind a flag/env var | ✅ Yes | Default-off, document the flag |
| New formats (WebM input?) | ✅ If small | Test with several real inputs before submitting |
| New processors (face_enhancer, frame_enhancer) | ✅ Yes | Add to the perf table; make optional |
| Mobile / desktop UI polish | ✅ Yes | Don't break the existing aesthetic |
| Tests / CI | ✅ Yes | None exist today, would love a smoke-test harness |
| Docker container | ⚠️ Maybe | Windows-Docker-with-CUDA is messy; design carefully |
| Cloud deployment recipes | ⚠️ Discuss first | Single-user local tool; security is a concern |
| Major refactors | ⚠️ Discuss first | Open an issue, scope it, agree on plan |
| Renaming public functions / endpoints | ❌ No | Back-compat with anyone forking |
| Removing the conda envs in favour of `uv` etc. | ❌ Not yet | Conda is stable, env conflicts are real |

---

## Commit style

```
short title in imperative voice (under 70 chars)

Why this change.

What was tried, what didn't work, lessons learned.

Measured impact (if perf):
  before:  X fps
  after:   Y fps  (+Z %)

Files changed: list the non-obvious ones.
```

Use `git commit -F .commit-msg.tmp` (PowerShell mangles `-m`
heredocs). There's a pattern in the existing commits — see
[../CLAUDE.md § PowerShell heredoc bug](../CLAUDE.md#6-powershell-heredoc--git-commit--m).

Examples worth modelling on:

```
git log --oneline -10
40ba7a1  Fix: produce a real MP4 (mobile + VLC compatible), not a fragmented one
d4fc024  Optimization #6: 4-stage pipeline (reader -> detect -> swap -> writer)
2a0e0dd  Multi-face swap: upload 1 or 2 source images, swap each lead with the matching one
0a966ce  det_size 640 -> 480 + Q_DEPTH 32 -> 64
660e7d1  Async reader thread + larger queues (Q_DEPTH 8 -> 32)
8735818  Async writer thread: decouple GPU pipeline from ffmpeg pipe writes
```

---

## What to test

### Always

- Smoke test in [HACKING.md § How to test a change](HACKING.md#how-to-test-a-change)
- Smoke-test on a fresh restart (don't rely on already-warm models)

### When changing the worker

- Pass frame 295 without crashing (a previous broken pipeline reliably
  crashed there with `list index out of range` — that's a regression
  marker)
- GPU avg util sampled over 6 s during a job is in the expected
  range for the resolution ([PERFORMANCE.md](PERFORMANCE.md))
- `webapp_jobs/<id>/ffmpeg.log` shows no warnings/errors

### When changing the UI

- Hard-refresh, not just reload
- Test with autoplay actually blocked (open in a fresh incognito tab)
- Verify "Click to unmute" still appears and works
- Verify all phase pills go green in order

### When changing the model loading

- Run `test-cuda-dlc.py` after restart, verify it prints
  `VERDICT: CUDA works`
- Check `webapp.log` for `[webapp] inswapper active providers:` —
  must contain `TensorrtExecutionProvider` (if installed) or
  `CUDAExecutionProvider` (never just `CPUExecutionProvider`)
- Sample GPU util during a job — should be >5 %

---

## Don't do

1. **Don't combine three optimisations in one commit.** When one
   breaks, you can't bisect. Land them in three small commits.
2. **Don't change behaviour without updating docs.** Especially
   USERGUIDE.md for user-visible changes, ARCHITECTURE.md for
   structural ones, CHANGELOG.md for everything.
3. **Don't commit personal photos or copyrighted videos.** The
   `.gitignore` excludes the standard places — don't paste them
   somewhere it doesn't.
4. **Don't add new dependencies casually.** `requirements-webapp.txt`
   is small for a reason — adding torch or transformers would
   double the install size.
5. **Don't bypass the verify-then-commit pattern.** Every change so
   far has been verified live (a job runs end-to-end) before commit.
   Don't break that habit.

---

## Reporting issues

Include:

1. The specific symptom
2. The phase the job got stuck on (or the error message in the
   viewer page)
3. Last 50 lines of `out/webapp.log`
4. Last 30 lines of `webapp_jobs/<id>/ffmpeg.log` (if streaming-
   related)
5. Browser DevTools console output (F12 → Console) if UI-related
6. Your environment: Windows version, GPU model, driver, Python /
   conda versions

---

## Code of conduct

Be kind, be specific, assume good faith.

This project is for entertainment and creative use. **Issues
proposing features that primarily enable non-consensual deepfakes,
sexual content involving real people, deceptive impersonation, or
content involving minors will be closed without discussion.**

The maintainer reserves the right to close PRs / issues that don't
fit the project's spirit.
