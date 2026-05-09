"use client";
import { useCallback, useEffect, useRef, useState } from "react";

interface Props {
  label: string;
  required?: boolean;
  optional?: boolean;
  hint?: string;
  accept: string;          // e.g. "image/*" or "video/*"
  multiple?: boolean;
  files: File[];
  onChange: (files: File[]) => void;
  isVideo?: boolean;
  small?: boolean;         // smaller variant for secondary slots
}

export function DropZone({
  label, required, optional, hint, accept, multiple,
  files, onChange, isVideo, small,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const previewUrls = files.map((f) => URL.createObjectURL(f));
  useEffect(() => () => previewUrls.forEach(URL.revokeObjectURL), [previewUrls]);

  const onPick = useCallback((picked: FileList | null) => {
    if (!picked) return;
    onChange(multiple ? Array.from(picked) : Array.from(picked).slice(0, 1));
  }, [multiple, onChange]);

  return (
    <label
      className={[
        "drop relative cursor-pointer rounded-3xl border-2 border-dashed",
        "border-white/10 bg-black/40 p-6 text-center transition",
        "hover:border-(--color-accent-1)/50 hover:-translate-y-0.5",
        over ? "!border-(--color-accent-1) !bg-[rgba(122,92,255,0.10)]" : "",
        files.length ? "!border-solid !border-(--color-good)/40 !bg-[rgba(82,214,163,0.05)]" : "",
        small ? "min-h-[140px] opacity-90 hover:opacity-100" : "min-h-[220px]",
        "flex flex-col items-center justify-center gap-2",
      ].join(" ")}
      onDragEnter={(e) => { e.preventDefault(); setOver(true); }}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={(e) => { e.preventDefault(); setOver(false); }}
      onDrop={(e) => {
        e.preventDefault(); setOver(false);
        onPick(e.dataTransfer.files);
      }}
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-(--color-accent-1)/30 bg-gradient-to-br from-(--color-accent-1)/20 to-(--color-accent-2)/20 text-(--color-accent-2)">
        {isVideo ? (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="23 7 16 12 23 17 23 7"></polygon>
            <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
          </svg>
        ) : (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="8" r="4"></circle>
            <path d="M4 21v-1a8 8 0 0 1 16 0v1"></path>
          </svg>
        )}
      </div>
      <div className="text-base font-semibold text-(--color-ink-0)">
        {label}
        {required && <span className="ml-1.5 text-xs font-medium text-[#ff8aa3]">(required)</span>}
        {optional && <span className="ml-1.5 text-xs font-medium text-(--color-ink-2)">(optional)</span>}
      </div>
      <div className="text-sm text-(--color-ink-2)">{hint || (isVideo ? "MP4, MOV, WebM" : "PNG, JPG · 1024 px+")}</div>

      {files.length > 0 && (
        <div className="mt-3 flex flex-wrap justify-center gap-2">
          {files.map((f, i) => (
            <div key={i} className="flex w-32 flex-col items-center gap-1 rounded-lg bg-black/60 p-2 text-[11px]">
              <div className="flex h-16 w-full items-center justify-center overflow-hidden rounded bg-black">
                {isVideo ? (
                  <video src={previewUrls[i]} className="h-full w-full object-cover" muted />
                ) : (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={previewUrls[i]} alt="" className="h-full w-full object-cover" />
                )}
              </div>
              <div className="line-clamp-1 break-all font-mono text-(--color-good)">
                {f.name}
              </div>
              <div className="font-mono text-(--color-ink-2)">
                {(f.size / 1024 / 1024).toFixed(1)} MB
              </div>
            </div>
          ))}
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        required={required && files.length === 0}
        className="absolute inset-0 cursor-pointer opacity-0"
        onChange={(e) => onPick(e.target.files)}
      />
    </label>
  );
}
