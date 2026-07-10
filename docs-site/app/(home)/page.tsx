import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="flex flex-col justify-center items-center text-center flex-1 gap-6 px-6 py-24">
      <h1 className="text-3xl font-bold">Hatchery</h1>
      <p className="max-w-xl text-fd-muted-foreground">
        An open-source, Tinker-compatible runtime for fine-tuning and
        post-training language models on infrastructure you control.
      </p>
      <div className="flex flex-wrap justify-center gap-4">
        <Link
          href="/docs/getting-started/quickstart"
          className="font-medium underline underline-offset-4"
        >
          Quickstart
        </Link>
        <Link
          href="/docs/getting-started/tinker-migration"
          className="font-medium underline underline-offset-4"
        >
          Migrating From Tinker
        </Link>
        <Link href="/docs/reference/endpoints" className="font-medium underline underline-offset-4">
          API Reference
        </Link>
        <Link href="/docs" className="font-medium underline underline-offset-4">
          All Docs
        </Link>
      </div>
    </div>
  );
}
