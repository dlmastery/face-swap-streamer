"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { HlsPlayer } from "@/components/HlsPlayer";
import {
  hlsPlaylistUrl, jobInlineFileUrl, jobDownloadUrl,
  subscribeJob, getJobStatus,
  type JobPhase, type JobStatus,
} from "@/lib/api";

const PHASE_ORDER: JobPhase[] = [
  "loading_models", "detecting_source", "finding_reference", "streaming", "finalising",
];

const PHASE_LABELS: Record<JobPhase, string> = {
  queued: "Queued",
  loading_models: "Loading models",
  detecting_source: "Detecting your face",
  finding_reference: "Finding target person",
  streaming: "Streaming live",
  finalising: "Finalising",
  done: "Done",
  error: "Error",
};

export default function JobViewer({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [s, setS] = useState<JobStatus | null>(null);

  useEffect(() => {
    let alive = true;
    // Initial snapshot via HTTP — handles the case where the job has already
    // gone terminal before WS connects (no future messages will arrive).
    getJobStatus(id).then((x) => { if (alive) setS(x); }).catch(() => {});
    const stop = subscribeJob(id, (x) => { if (alive) setS(x); });
    return () => { alive = false; stop(); };
  }, [id]);

  const phase = s?.phase ?? "queued";
  const idx = PHASE_ORDER.indexOf(phase as JobPhase);
  const isStreaming = phase === "streaming" || phase === "finalising" || phase === "done";
  const isDone = phase === "done";
  const isError = phase === "error";

  const progress = s && s.total_frames > 0
    ? (100 * s.current_frame) / s.total_frames
    : 0;

  return (
    <main className="mx-auto w-full max-w-6xl px-6 py-8 pb-24">
      <div className="mb-4 flex items-center justify-between text-sm">
        <span className="font-mono text-(--color-ink-1)">job · {id}</span>
        <Link href="/" className="text-(--color-ink-1) opacity-80 transition hover:text-(--color-accent-2) hover:opacity-100">
          ← new swap
        </Link>
      </div>

      {isStreaming ? (
        <HlsPlayer
          src={hlsPlaylistUrl(id)}
          staticSrc={jobInlineFileUrl(id)}
          useStaticOnDone={isDone}
        />
      ) : (
        <div className="relative flex aspect-video w-full flex-col items-center justify-center gap-3 overflow-hidden rounded-2xl border border-white/5 bg-[radial-gradient(800px_500px_at_50%_30%,rgba(122,92,255,0.10),transparent_60%)] text-center">
          <div className="ring" />
          <div className="text-lg font-semibold">{PHASE_LABELS[phase as JobPhase] || phase}</div>
          <div className="max-w-md text-sm text-(--color-ink-1)">
            {s?.message ?? "Initialising…"}
          </div>
          <div className="mt-4 flex flex-wrap justify-center gap-2 font-mono text-xs">
            {PHASE_ORDER.map((k, i) => (
              <span
                key={k}
                className={`rounded-full border px-3 py-1.5 transition ${
                  i < idx
                    ? "border-(--color-good)/30 bg-(--color-good)/[0.08] text-(--color-good)"
                    : i === idx
                      ? "border-(--color-accent-1) bg-(--color-accent-1)/[0.15] text-(--color-ink-0) shadow-[0_0_24px_rgba(122,92,255,0.25)]"
                      : "border-transparent bg-white/[0.04] text-(--color-ink-2)"
                }`}
              >
                {k.replace("_", " ")}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="mt-5 h-1.5 w-full overflow-hidden rounded-sm bg-white/[0.06]">
        <div
          className="h-full bg-gradient-to-r from-(--color-accent-1) to-(--color-accent-2) transition-[width] duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>

      <div className="mt-2 flex flex-wrap gap-6 font-mono text-sm text-(--color-ink-2)">
        <span>progress <b className="font-medium text-(--color-ink-0)">{s?.current_frame ?? 0} / {s?.total_frames ?? 0}</b></span>
        <span>fps <b className="font-medium text-(--color-ink-0)">{s?.proc_fps?.toFixed(1) ?? "–"}</b></span>
        <span>swaps <b className="font-medium text-(--color-ink-0)">{s?.swap_count ?? 0}</b></span>
        {s?.sources?.map((src, i) => (
          <span key={i}>
            src{i + 1} <b className="font-medium text-(--color-ink-0)">{src.gender}/{src.age}</b>
            {src.ref_frame >= 0 && <span> (f{src.ref_frame}, {src.ref_votes}/{src.ref_pool})</span>}
          </span>
        ))}
      </div>

      {isDone && (
        <div className="mt-6 flex flex-col gap-3 rounded-2xl border border-(--color-good)/25 bg-gradient-to-br from-(--color-good)/[0.08] to-(--color-accent-2)/[0.08] p-7 shadow-[0_20px_50px_rgba(0,0,0,0.4)]">
          <h2 className="bg-gradient-to-br from-(--color-good) to-(--color-accent-2) bg-clip-text text-2xl font-bold text-transparent">
            Your swap is ready
          </h2>
          <p className="text-sm text-(--color-ink-1)">
            Audio is included. The video above is the final result — controls let you scrub, replay, and full-screen.
          </p>
          <a
            href={jobDownloadUrl(id)}
            download
            className="inline-flex w-fit items-center gap-2 rounded-xl bg-gradient-to-br from-(--color-accent-1) to-(--color-accent-2) px-5 py-3 text-sm font-semibold text-white shadow-[0_12px_30px_rgba(122,92,255,0.35)] transition hover:-translate-y-0.5"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            Download MP4 (with audio)
          </a>
        </div>
      )}

      {isError && (
        <div className="mt-5 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-200">
          Job failed: {s?.message || s?.error}
        </div>
      )}
    </main>
  );
}
