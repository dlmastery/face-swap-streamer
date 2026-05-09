import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const inter = Inter({ variable: "--font-inter", subsets: ["latin"] });
const mono = JetBrains_Mono({ variable: "--font-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Faceswap streamer",
  description:
    "Live face-swap web app: upload photo + video, watch the swap stream "
    + "to your browser with synchronised audio while it processes.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${inter.variable} ${mono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col">
        <div className="aurora"><div className="blob" /></div>

        <header className="sticky top-0 z-10 flex items-center justify-between border-b border-white/[0.07] bg-[rgba(5,6,12,0.45)] px-6 py-4 backdrop-blur">
          <Link href="/" className="flex items-center gap-2 font-bold tracking-tight">
            <span className="block h-2.5 w-2.5 rounded-full bg-gradient-to-br from-(--color-accent-1) to-(--color-accent-3) shadow-[0_0_18px_var(--color-accent-1)]" />
            Faceswap
          </Link>
          <a
            href="https://github.com/dlmastery/face-swap-streamer"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-(--color-ink-1) transition hover:text-(--color-accent-2)"
          >
            github
          </a>
        </header>

        {children}
      </body>
    </html>
  );
}
