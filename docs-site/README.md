# Hatchery Docs Site

Public docs site for Hatchery, built with [Fumadocs](https://fumadocs.dev) on
Next.js. It renders the existing `content/docs/` MDX tree (one level up, at
the repo root) — content is not duplicated or forked into this directory.

## Why Fumadocs, Not Mintlify

`content/docs/` already ships navigation metadata for both frameworks: a
root `docs.json` (Mintlify) and per-folder `meta.json` files (Fumadocs). Both
could technically read the existing MDX with no content rewrites. Fumadocs
was chosen because:

- **It can actually be self-hosted with a real static build.** Fumadocs is a
  plain Next.js app; `next build` with `output: 'export'` produces static
  HTML you can serve from anything (see below). Mintlify's tooling is built
  around its own hosted platform and a local `mintlify dev` preview server —
  there's no equivalent standalone static export that doesn't depend on
  Mintlify's infrastructure, which conflicts with the "local dev + static
  build both work" requirement for this site.
- **The `meta.json` files already match Fumadocs' exact convention.** Each
  folder's `meta.json` (`title` + ordered `pages` array, folder names
  expanding into nested groups) is precisely the shape
  `fumadocs-core`'s source loader expects — no restructuring needed, only a
  `source.config.ts` pointing `dir` at `../content/docs`.
- **Frontmatter is already compatible.** Every page's `title`/`description`
  frontmatter parses under Fumadocs' default page schema unchanged.

`docs.json` is left in place and kept in sync for navigation (grouping/order)
so a future switch to Mintlify, or a Mintlify-based preview, stays possible
without re-deriving the nav structure from scratch.

## Structure

```
docs-site/
  app/                  Next.js app router pages/layouts
  components/           MDX component overrides, provider wiring
  lib/                  source loader, shared nav/site config
  source.config.ts       fumadocs-mdx collection config (dir: ../content/docs)
  next.config.mjs        output: 'export' (static site), MDX plugin
```

Content itself lives in `../content/docs` and is untouched by this
directory except where a page didn't exist yet (see
`../content/docs/getting-started/tinker-migration.mdx`,
`../content/docs/cookbook/`, `../content/docs/pricing.mdx`, added to fill
out the required nav) or where `meta.json`/`docs.json` ordering was updated
to surface Quickstart and the Tinker migration guide first.

## Local Dev

```bash
cd docs-site
npm install
npm run dev
```

Serves at `http://localhost:3000` with hot reload over the MDX in
`../content/docs`.

## Static Build

```bash
cd docs-site
npm install
npm run build
```

Outputs a fully static site to `docs-site/out/`. Preview it locally with any
static file server, for example:

```bash
npm run start   # serves out/ via `serve`
```

Deploy `out/` to any static host (S3+CloudFront, Cloudflare Pages, GitHub
Pages, Vercel static hosting, nginx, etc.) — there is no Node.js server
required at runtime.

## Search

Static export has no backend to serve a search index. Search is disabled
(`search: { enabled: false }` in `components/provider.tsx`) rather than
half-wired. Revisit with fumadocs-core's static Orama search
(`fumadocs-core/search/client/orama-static`) if the docs set grows enough to
need in-site search.

## Placeholders To Revisit

- `lib/shared.ts`'s `statusPageUrl` (`https://status.hatchery.ai`) is a
  placeholder until the hosted platform publishes a real status page.
- `../content/docs/pricing.mdx` is a placeholder page — hosted-gateway
  pricing hasn't been published yet.
