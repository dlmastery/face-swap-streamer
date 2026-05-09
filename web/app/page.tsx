"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { DropZone } from "@/components/DropZone";
import { uploadBatch, uploadSingle } from "@/lib/api";

export default function Home() {
  const router = useRouter();
  const [sources, setSources] = useState<File[]>([]);
  const [sources2, setSources2] = useState<File[]>([]);
  const [targets, setTargets] = useState<File[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (sources.length === 0) {
      setError("Face #1 is required");
      return;
    }
    if (targets.length === 0) {
      setError("At least one target video is required");
      return;
    }
    setSubmitting(true);
    try {
      const allSources = [...sources, ...sources2];
      if (targets.length === 1) {
        const res = await uploadSingle(allSources, targets[0]);
        router.push(`/jobs/${res.id}`);
      } else {
        const res = await uploadBatch(allSources, targets);
        router.push(`/batches/${res.id}`);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "upload failed");
      setSubmitting(false);
    }
  };

  return (
    <main className="mx-auto w-full max-w-6xl px-6 py-16 pb-24">
      <section className="mb-14 text-center">
        <span className="mb-5 inline-block rounded-full border border-(--color-accent-1)/30 bg-(--color-accent-1)/10 px-4 py-1.5 text-xs font-medium uppercase tracking-wider text-[#c4b3ff]">
          Live face-swap streaming
        </span>
        <h1 className="bg-gradient-to-br from-white via-(--color-ink-1) to-(--color-accent-1) bg-clip-text text-5xl font-extrabold leading-[1.05] tracking-[-0.03em] text-transparent md:text-6xl">
          Your face, in any video.
          <br />
          Streamed live to your browser.
        </h1>
        <p className="mx-auto mt-4 max-w-2xl text-base leading-relaxed text-(--color-ink-1) md:text-lg">
          Drop in a photo of yourself (or two for a duet) and one or more videos. We auto-detect
          gender, lock onto each matching person, and stream the swap with synchronised audio
          frame by frame, while it&apos;s still being processed. Multiple videos run as a batch and
          download as a single zip.
        </p>
      </section>

      <form onSubmit={onSubmit} className="mx-auto max-w-5xl rounded-3xl border border-white/[0.07] bg-gradient-to-b from-[rgba(20,26,42,0.65)] to-[rgba(13,16,28,0.8)] p-8 shadow-[0_30px_80px_rgba(0,0,0,0.45),inset_0_1px_0_rgba(255,255,255,0.06)] backdrop-blur-xl">
        <div className="grid items-stretch gap-5 md:grid-cols-[1fr_72px_1.4fr]">
          <div className="flex flex-col gap-3">
            <DropZone
              label="Face #1"
              required
              accept="image/*"
              files={sources}
              onChange={setSources}
              hint="PNG, JPG · 1024 px+ recommended"
            />
            <DropZone
              label="Face #2"
              optional
              accept="image/*"
              files={sources2}
              onChange={setSources2}
              hint="For duets — swap both leads"
              small
            />
          </div>

          <div className="hidden items-center justify-center text-(--color-accent-1) md:flex">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={{ animation: "blink 2.4s ease-in-out infinite" }}>
              <line x1="5" y1="12" x2="19" y2="12"></line>
              <polyline points="12 5 19 12 12 19"></polyline>
            </svg>
          </div>

          <DropZone
            label="Target video(s)"
            required
            multiple
            isVideo
            accept="video/*"
            files={targets}
            onChange={setTargets}
            hint="Drop one for a single swap, several for a batch"
          />
        </div>

        {error && (
          <div className="mt-5 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
            {error}
          </div>
        )}

        <div className="mt-7 flex flex-wrap items-center gap-4">
          <button
            type="submit"
            disabled={submitting}
            className="flex-1 min-w-[200px] rounded-xl bg-gradient-to-br from-(--color-accent-1) to-(--color-accent-2) px-6 py-3.5 text-base font-semibold text-white shadow-[0_14px_30px_rgba(122,92,255,0.35)] transition hover:-translate-y-0.5 hover:shadow-[0_18px_40px_rgba(122,92,255,0.45)] disabled:cursor-wait disabled:opacity-60"
          >
            {submitting
              ? "Uploading…"
              : targets.length > 1
                ? `Start batch swap (${targets.length} videos)`
                : "Start live swap"}
          </button>
          <span className="text-xs text-(--color-ink-2)">
            First run loads models (~30 s). After that, every job is fast.
          </span>
        </div>
      </form>

      <section className="mx-auto mt-12 grid max-w-5xl gap-4 md:grid-cols-3">
        {[
          { title: "Auto gender + reference lock", body: "Detects each face's gender from the source, scans the video, locks the swap onto the matching person — never the other co-star." },
          { title: "HLS streaming with audio", body: "Browser plays the swap with the original song's audio while it's still being processed — no waiting for the full render." },
          { title: "Batch + zip download", body: "Drop several videos at once. They process sequentially on the GPU and you can download all the finished MP4s as a single zip." },
        ].map((f) => (
          <div key={f.title} className="rounded-2xl border border-white/[0.07] bg-[rgba(13,16,28,0.5)] p-5">
            <h3 className="text-base font-semibold">{f.title}</h3>
            <p className="mt-1 text-sm leading-relaxed text-(--color-ink-2)">{f.body}</p>
          </div>
        ))}
      </section>

      <footer className="mt-16 text-center text-xs text-(--color-ink-2)">
        local · GPU-accelerated via CUDA · models cached after first run
      </footer>
    </main>
  );
}
