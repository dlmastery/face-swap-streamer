"use client";
import { useEffect, useRef, useState } from "react";
import Hls from "hls.js";

interface Props {
  /** HLS playlist URL. */
  src: string;
  /** Poster / fallback static MP4 URL once the run has finished. If
   *  provided and `useStaticOnDone` is true, switches to a regular
   *  `<video src=...>` instead of HLS. */
  staticSrc?: string;
  useStaticOnDone?: boolean;
  /** Seconds buffered before pressing play. Higher = smoother but later start. */
  preBufferSeconds?: number;
  /** Seconds buffered before resuming after a stall. */
  reBufferSeconds?: number;
}

/**
 * HLS live-or-VOD player with:
 *   - 15 s pre-buffer before pressing play (avoids stuttering on slower-than-realtime swap)
 *   - re-buffer recovery on stall events
 *   - autoplay muted + click-to-unmute pill (browsers block autoplay-with-sound)
 *   - any-click-to-rescue handler if Chrome refuses autoplay outright
 */
export function HlsPlayer({
  src,
  staticSrc,
  useStaticOnDone,
  preBufferSeconds = 15,
  reBufferSeconds = 8,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const hlsRef = useRef<Hls | null>(null);
  const [bufMessage, setBufMessage] = useState<string>("Loading…");
  const [showOverlay, setShowOverlay] = useState(false);
  const [playStarted, setPlayStarted] = useState(false);

  // Bind / re-bind whenever `src` changes
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;

    if (useStaticOnDone && staticSrc) {
      hlsRef.current?.destroy();
      hlsRef.current = null;
      v.src = staticSrc;
      v.muted = false;
      setBufMessage("");
      setShowOverlay(false);
      setPlayStarted(true);
      return;
    }

    if (Hls.isSupported()) {
      const hls = new Hls({
        liveSyncDuration: preBufferSeconds,
        liveMaxLatencyDuration: 60,
        maxBufferLength: 60,
        maxMaxBufferLength: 120,
        backBufferLength: 90,
        manifestLoadingMaxRetry: 60,
        manifestLoadingRetryDelay: 800,
        levelLoadingMaxRetry: 60,
        levelLoadingRetryDelay: 800,
        fragLoadingMaxRetry: 60,
        fragLoadingRetryDelay: 800,
      });
      hls.loadSource(src);
      hls.attachMedia(v);
      hlsRef.current = hls;

      const tryStart = () => {
        if (playStarted) return;
        const buffered = v.buffered.length
          ? v.buffered.end(v.buffered.length - 1) - v.currentTime : 0;
        setBufMessage(`Buffering ${buffered.toFixed(1)} / ${preBufferSeconds}s before starting…`);
        if (buffered < preBufferSeconds) return;
        v.muted = true;
        v.play()
          .then(() => {
            setPlayStarted(true);
            setBufMessage("");
            setShowOverlay(true);  // unmute pill
          })
          .catch(() => setTimeout(tryStart, 1000));
      };

      hls.on(Hls.Events.BUFFER_APPENDED, tryStart);
      hls.on(Hls.Events.ERROR, (_e, data) => {
        if (data.fatal) console.warn("hls fatal", data);
      });
    } else if (v.canPlayType("application/vnd.apple.mpegurl")) {
      v.src = src;
      v.play().catch(() => {});
    } else {
      setBufMessage("Your browser doesn't support HLS.");
    }

    // Stall handling
    const onWaiting = () => {
      if (playStarted) {
        const ahead = v.buffered.length
          ? v.buffered.end(v.buffered.length - 1) - v.currentTime : 0;
        setBufMessage(`Re-buffering ${ahead.toFixed(1)} / ${reBufferSeconds}s…`);
      }
    };
    const onPlaying = () => setBufMessage("");
    v.addEventListener("waiting", onWaiting);
    v.addEventListener("playing", onPlaying);

    // Click-anywhere autoplay rescue
    const rescue = () => {
      const buffered = v.buffered.length
        ? v.buffered.end(v.buffered.length - 1) - v.currentTime : 0;
      if (!playStarted && buffered >= 1) {
        v.muted = true;
        v.play().then(() => {
          setPlayStarted(true);
          setShowOverlay(true);
          setBufMessage("");
        }).catch(() => {});
      }
    };
    document.addEventListener("click", rescue);

    return () => {
      v.removeEventListener("waiting", onWaiting);
      v.removeEventListener("playing", onPlaying);
      document.removeEventListener("click", rescue);
      hlsRef.current?.destroy();
      hlsRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src, staticSrc, useStaticOnDone]);

  const onUnmute = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (videoRef.current) {
      videoRef.current.muted = false;
      videoRef.current.volume = 1;
    }
    setShowOverlay(false);
  };

  return (
    <div className="relative aspect-video w-full overflow-hidden rounded-2xl border border-white/5 bg-black shadow-[0_30px_80px_rgba(0,0,0,0.5)]">
      <video
        ref={videoRef}
        playsInline
        controls
        muted
        autoPlay
        className="h-full w-full bg-black object-contain"
      />

      {bufMessage && (
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-3 bg-[radial-gradient(800px_500px_at_50%_30%,rgba(122,92,255,0.10),transparent_60%)] text-center">
          <div className="ring" />
          <div className="text-sm text-(--color-ink-1)">{bufMessage}</div>
        </div>
      )}

      {showOverlay && (
        <button
          onClick={onUnmute}
          className="absolute bottom-4 left-4 inline-flex items-center gap-2 rounded-full border border-(--color-accent-1)/50 bg-[#141a2aee] px-4 py-2 text-sm font-semibold text-(--color-ink-0) shadow-[0_14px_40px_rgba(0,0,0,0.6),0_0_0_4px_rgba(122,92,255,0.15)] transition hover:-translate-y-0.5"
          style={{ animation: "pulse-glow 2.4s ease-in-out infinite" }}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
            <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
            <path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>
          </svg>
          Click to unmute
        </button>
      )}
    </div>
  );
}
