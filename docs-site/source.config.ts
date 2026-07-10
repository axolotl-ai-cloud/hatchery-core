import { defineConfig, defineDocs } from 'fumadocs-mdx/config';

// Reads content from the repo-root `content/docs` tree (shared with the
// Mintlify-compatible `docs.json` navigation) instead of duplicating it
// under docs-site/. See README.md for the framework choice rationale.
export const docs = defineDocs({
  dir: '../content/docs',
});

export default defineConfig({
  mdxOptions: {},
});
