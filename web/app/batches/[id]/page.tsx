"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import {
  batchDownloadZipUrl, getBatchStatus, subscribeBatch, jobDownloadUrl,
  type BatchStatus, type JobPhase,
} from "@/lib/api";

const PHASE_LABELS: Record<JobPhase, string> = {
  queued: "Queued",
  loading_models: "Loading models",
  detecting_source: "Detecting source",
  finding_reference: "Finding reference",
  streaming: "Streaming",
  finalising: "Finalising",
  done: "Done",
  error: "Error",
};

export default function BatchViewer({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [s, setS] = useState<BatchStatus | null>(null);

  useEffect(() => {
    let alive = true;
    getBatchStatus(id).then((x) => { if (alive) setS(x); }).catch(() => {});
    const stop = subscribeBatch(id, (x) => { if (alive) setS(x); });
    return () => { alive = false; stop(); };
  }, [id]);

  const isDone = s?.phase === "done";
  const hasErrors = (s?.error_videos ?? 0) > 0;
  const readyCount = s?.done_videos ?? 0;
  const canDownloadAny = readyCount > 0;

  return (
    <main className="mx-auto w-full max-w-6xl px-6 py-8 pb-24">
      <div className="mb-4 flex items-center justify-between text-sm">
        <span className="font-mono text-(--color-ink-1)">batch · {id}</span>
        <Link href="/" className="text-(--color-ink-1) opacity-80 transition hover:text-(--color-accent-2) hover:opacity-100">
          ← new batch
        </Link>
      </div>

      <header className="mb-6 rounded-2xl border border-white/5 bg-[rgba(13,16,28,0.5)] p-6">
        <div className="flex flex-wrap items-baseline gap-3">
          <h1 className="text-2xl font-bold">
            {s?.total_videos ?? 0} video{s?.total_videos === 1 ? "" : "s"}
          </h1>
          <span className="font-mono text-sm text-(--color-ink-2)">
            {s?.done_videos ?? 0} done · {s?.error_videos ?? 0} failed · phase: {s?.phase ?? "queued"}
          </span>
        </div>
        <p className="mt-2 text-sm text-(--color-ink-1)">{s?.message ?? "Initialising…"}</p>
        {s?.sources && s.sources.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2 font-mono text-xs text-(--color-ink-2)">
            {s.sources.map((src, i) => (
              <span key={i} className="rounded-full bg-white/[0.05] px-3 py-1">
                src{i + 1} <b className="text-(--color-ink-0)">{src.gender || "?"}/{src.age || "?"}</b>
              </span>
            ))}
          </div>
        )}

        {canDownloadAny && (
          <a
            href={batchDownloadZipUrl(id)}
            download
            className="mt-5 inline-flex w-fit items-center gap-2 rounded-xl bg-gradient-to-br from-(--color-accent-1) to-(--color-accent-2) px-5 py-3 text-sm font-semibold text-white shadow-[0_12px_30px_rgba(122,92,255,0.35)] transition hover:-translate-y-0.5"
            title={isDone
              ? `Download all ${readyCount} swapped MP4s as a single zip`
              : `Download the ${readyCount} ready so far as a zip (rest will appear as they finish)`}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            {isDone
              ? `Download all ${readyCount} as zip`
              : `Download ready ${readyCount} of ${s?.total_videos ?? 0} as zip`}
          </a>
        )}
        {hasErrors && (
          <div className="mt-3 inline-block rounded-lg bg-red-500/10 px-3 py-1.5 text-sm text-red-200">
            {s?.error_videos} video{s?.error_videos === 1 ? "" : "s"} failed — see per-video tiles below
          </div>
        )}
      </header>

      <ul className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
        {(s?.jobs ?? []).map((j) => {
          const isJobDone = j.phase === "done";
          const isJobError = j.phase === "error";
          return (
            <li key={j.id} className="rounded-2xl border border-white/5 bg-[rgba(13,16,28,0.5)] p-5">
              <Link href={`/jobs/${j.id}`} className="block">
                <div className="mb-2 line-clamp-1 break-all font-mono text-sm text-(--color-ink-0)">
                  {j.target_filename}
                </div>
                <div className="text-xs text-(--color-ink-2)">
                  {PHASE_LABELS[j.phase] || j.phase}
                  {j.proc_fps > 0 && <> · {j.proc_fps.toFixed(1)} fps</>}
                </div>
                <div className="mt-3 h-1 overflow-hidden rounded-sm bg-white/[0.06]">
                  <div
                    className={`h-full transition-[width] duration-300 ${
                      isJobError
                        ? "bg-red-500/70"
                        : isJobDone
                          ? "bg-(--color-good)"
                          : "bg-gradient-to-r from-(--color-accent-1) to-(--color-accent-2)"
                    }`}
                    style={{ width: `${isJobDone ? 100 : j.progress_pct}%` }}
                  />
                </div>
                {isJobError && (
                  <div className="mt-2 line-clamp-2 break-all text-xs text-red-200/80">
                    {j.error}
                  </div>
                )}
              </Link>
              {isJobDone && (
                <a
                  href={jobDownloadUrl(j.id)}
                  download
                  className="mt-3 block rounded-lg bg-(--color-accent-2)/[0.15] px-3 py-2 text-center text-xs font-medium text-[#9bbcff] transition hover:bg-(--color-accent-2)/25 hover:text-white"
                  onClick={(e) => e.stopPropagation()}
                >
                  Download this MP4
                </a>
              )}
            </li>
          );
        })}
      </ul>
    </main>
  );
}
