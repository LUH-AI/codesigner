"""Build the assistant's library: `manage.py assistant_knowledge build`.

Fetches each paper in assistant/knowledge/literature.toml from its open-access
source, splits it into passages, embeds them through the embedding endpoint
(`ASSISTANT_EMBED_*`) unless `--no-embed`, and writes the index to
`ASSISTANT_KNOWLEDGE_DIR`. `--only KEY ...` fetches just those papers again.

`ensure` builds only what is needed: nothing while the assistant is off or the
index matches the paper list and the embedding model, else a build on the
worker's queue, or here with `--now`. The worker runs it on starting.
"""

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Build the assistant's library of papers."

    def add_arguments(self, parser):
        parser.add_argument("action", choices=["build", "status", "ensure"])
        parser.add_argument("--only", nargs="+", default=None, help="Keys to fetch again.")
        parser.add_argument("--no-embed", action="store_true",
                            help="Build for search by words alone.")
        parser.add_argument("--now", action="store_true",
                            help="With ensure: build here rather than on the queue.")

    def handle(self, *args, **options):
        import json

        from assistant import library
        from assistant.llm import EmbeddingClient

        if options["action"] == "ensure":
            return self.ensure(now=options["now"])
        if options["action"] == "status":
            built = library.index_dir() / "built.json"
            if not built.exists():
                raise CommandError(f"No library at {library.index_dir()}.")
            self.stdout.write(built.read_text())
            self.warn_failures(json.loads(built.read_text()))
            return
        embedder = None if options["no_embed"] else EmbeddingClient.from_settings()
        if embedder is None and not options["no_embed"]:
            raise CommandError("No embedding endpoint (ASSISTANT_EMBED_BASE_URL); pass --no-embed "
                               "to build for search by words alone.")
        only = set(options["only"]) if options["only"] else None
        unknown = (only or set()) - {p["key"] for p in library.papers()}
        if unknown:
            raise CommandError(f"Not in the list: {sorted(unknown)}")
        built = library.build(embedder=embedder, only=only, progress=self.stdout.write)
        if built is not None:
            self.stdout.write(json.dumps(built, indent=2))
            self.warn_failures(built)

    def ensure(self, now: bool):
        from django.conf import settings

        from assistant import library
        from assistant.llm import EmbeddingClient

        if not settings.ASSISTANT_ENABLED:
            self.stdout.write("The assistant is off; no library is needed.")
            return
        embedder = EmbeddingClient.from_settings()
        reason = library.stale(getattr(embedder, "model", None))
        if not reason:
            self.stdout.write(f"The library at {library.index_dir()} is current.")
            self.warn_failures(library.built())
            return
        if now:
            self.stdout.write(f"Building the library: {reason}.")
            self.warn_failures(library.build(embedder=embedder, progress=self.stdout.write))
            return
        from assistant.tasks import build_library_task

        build_library_task()
        self.stdout.write(f"Library build queued: {reason}.")

    def warn_failures(self, record):
        """`warn_failures` names each paper the last build could not fetch,
        which the library knows only by its summary until a build fetches it."""
        failed = (record or {}).get("failed") or {}
        if not failed:
            return
        from assistant.sources import papers, source_of

        where = {p["key"]: source_of(p)[0] for p in papers()}
        self.stderr.write(self.style.WARNING(
            f"{len(failed)} paper(s) of the library could not be fetched and are known only "
            f"by their summary:"))
        for key, reason in sorted(failed.items()):
            self.stderr.write(self.style.WARNING(f"  {key}  {where.get(key, '')}  {reason}"))
