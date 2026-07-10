'use client';
import { RootProvider } from 'fumadocs-ui/provider/next';
import type { ReactNode } from 'react';

// Search is disabled for now — no server to back a search index in the
// static export. Revisit with fumadocs-core's static Orama search if the
// docs set grows large enough to need it.
export function Provider({ children }: { children: ReactNode }) {
  return <RootProvider search={{ enabled: false }}>{children}</RootProvider>;
}
