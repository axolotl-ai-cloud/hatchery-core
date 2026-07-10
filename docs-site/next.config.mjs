import { fileURLToPath } from 'node:url';
import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  output: 'export',
  reactStrictMode: true,
  // Content lives in `../content/docs` (shared with the Mintlify-compatible
  // docs.json nav) instead of being duplicated under docs-site/. Turbopack's
  // default root inference stops at this directory, so it refuses to resolve
  // MDX/meta.json imports one level up unless we widen the root explicitly.
  turbopack: {
    root: fileURLToPath(new URL('..', import.meta.url)),
  },
};

export default withMDX(config);
