/**
 * Typed API client. Mirrors server/schemas.py — keep them in sync.
 * In dev, requests go through Next.js rewrites to the FastAPI server.
 * In prod (static export), requests hit the FastAPI server directly
 * because the static frontend is mounted under FastAPI's /.
 */

export type JobPhase =
  | "queued"
  | "loading_models"
  | "detecting_source"
  | "finding_reference"
  | "streaming"
  | "finalising"
  | "done"
  | "error";

export type Gender = "" | "M" | "F";

export interface SourceInfo {
  gender: Gender;
  age: number;
  ref_frame: number;
  ref_votes: number;
  ref_pool: number;
}

export interface JobStatus {
  id: string;
  phase: JobPhase;
  message: string;
  error: string;
  sources: SourceInfo[];
  detected_gender: Gender;
  detected_age: number;
  ref_frame: number;
  ref_votes: number;
  ref_pool: number;
  width: number;
  height: number;
  fps: number;
  total_frames: number;
  current_frame: number;
  swap_count: number;
  proc_fps: number;
  started: number;
  finished: number;
}

export interface JobCreated {
  id: string;
  status_url: string;
  ws_url: string;
  hls_url: string;
  download_url: string;
}

export type BatchPhase = "queued" | "processing" | "done" | "error";

export interface BatchJobSummary {
  id: string;
  target_filename: string;
  phase: JobPhase;
  progress_pct: number;
  proc_fps: number;
  error: string;
}

export interface BatchStatus {
  id: string;
  phase: BatchPhase;
  message: string;
  sources: SourceInfo[];
  jobs: BatchJobSummary[];
  total_videos: number;
  done_videos: number;
  error_videos: number;
  started: number;
  finished: number;
}

export interface BatchCreated {
  id: string;
  status_url: string;
  ws_url: string;
  download_zip_url: string;
  jobs: string[];
}

/** Base URL for the FastAPI backend. In dev, point straight at :8081 so
 * we bypass the Next.js dev proxy (which has a 10 MB body limit and chokes
 * on real video uploads). In production, the Next.js static build is
 * served from FastAPI's `/`, so empty string == same origin works. */
const apiBase =
  (typeof process !== "undefined" && process.env.NEXT_PUBLIC_API_URL)
  || (typeof window !== "undefined" && window.location.port === "3000"
      ? `${window.location.protocol}//${window.location.hostname}:8081`
      : "");

export async function uploadSingle(
  sources: File[],
  target: File,
): Promise<JobCreated> {
  const fd = new FormData();
  for (const f of sources) fd.append("source", f);
  fd.append("target", target);
  const res = await fetch(`${apiBase}/api/jobs`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(await res.text() || `upload failed (${res.status})`);
  return res.json();
}

export async function uploadBatch(
  sources: File[],
  targets: File[],
): Promise<BatchCreated> {
  const fd = new FormData();
  for (const f of sources) fd.append("source", f);
  for (const f of targets) fd.append("target", f);
  const res = await fetch(`${apiBase}/api/batches`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(await res.text() || `upload failed (${res.status})`);
  return res.json();
}

export async function getJobStatus(id: string): Promise<JobStatus> {
  const res = await fetch(`${apiBase}/api/jobs/${id}/status`);
  if (!res.ok) throw new Error(`job status failed (${res.status})`);
  return res.json();
}

export async function getBatchStatus(id: string): Promise<BatchStatus> {
  const res = await fetch(`${apiBase}/api/batches/${id}/status`);
  if (!res.ok) throw new Error(`batch status failed (${res.status})`);
  return res.json();
}

/** Open a typed WebSocket for live job updates. Auto-reconnects once on
 * unclean close. Returns a teardown function. */
export function subscribeJob(
  id: string,
  onMessage: (s: JobStatus) => void,
  onError?: (e: Event) => void,
): () => void {
  return _subscribe(`/api/jobs/${id}/ws`, onMessage, onError);
}

export function subscribeBatch(
  id: string,
  onMessage: (s: BatchStatus) => void,
  onError?: (e: Event) => void,
): () => void {
  return _subscribe(`/api/batches/${id}/ws`, onMessage, onError);
}

function _subscribe<T>(path: string, onMessage: (m: T) => void, onError?: (e: Event) => void) {
  let closedByUs = false;
  let ws: WebSocket | null = null;
  // Use the same apiBase so the WebSocket connects directly to FastAPI
  // (no Next.js proxy, which doesn't forward WebSocket upgrade properly).
  const wsBase =
    apiBase ||
    (typeof window !== "undefined" ? window.location.origin : "");
  const wsUrl = wsBase.replace(/^http/, "ws") + path;

  const open = () => {
    ws = new WebSocket(wsUrl);
    ws.onmessage = (ev) => {
      try { onMessage(JSON.parse(ev.data)); } catch { /* ignore malformed */ }
    };
    ws.onerror = (e) => { onError?.(e); };
    ws.onclose = () => {
      if (!closedByUs) setTimeout(open, 1000);  // single auto-reconnect
    };
  };
  open();

  return () => { closedByUs = true; ws?.close(); };
}

export const hlsPlaylistUrl = (jobId: string) =>
  `${apiBase}/api/jobs/${jobId}/hls/playlist.m3u8`;

export const jobDownloadUrl = (jobId: string) =>
  `${apiBase}/api/jobs/${jobId}/download`;

export const jobInlineFileUrl = (jobId: string) =>
  `${apiBase}/api/jobs/${jobId}/file`;

export const batchDownloadZipUrl = (batchId: string) =>
  `${apiBase}/api/batches/${batchId}/download`;
