/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_GEMINI_API: process.env.GEMINI_API,
  },
}
module.exports = nextConfig
