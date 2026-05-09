import type { NextConfig } from "next";

/**
 * Production builds export to a fully static `out/` dir which the FastAPI
 * server mounts at `/`. In dev (`npm run dev` on :3000) we proxy /api and
 * /healthz through to the FastAPI server on :8081 — keeps the frontend at
 * the same origin so cookies, WebSockets, and CORS all just work.
 */
const FASTAPI_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8081";

const nextConfig: NextConfig = {
  output: process.env.NEXT_BUILD_STATIC ? "export" : undefined,
  distDir: process.env.NEXT_BUILD_STATIC ? "../server/static/dist" : ".next",
  trailingSlash: true,
  images: { unoptimized: true },

  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${FASTAPI_URL}/api/:path*` },
      { source: "/healthz", destination: `${FASTAPI_URL}/healthz` },
    ];
  },
};

export default nextConfig;
