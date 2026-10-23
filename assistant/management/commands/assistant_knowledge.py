"""Build the assistant's library: `manage.py assistant_knowledge build`.

Fetches each paper in assistant/knowledge/literature.toml from its open-access
source, splits it into passages, embeds them through the embedding endpoint
(`ASSISTANT_EMBED_*`) unless `--no-embed`, and writes the index to
`ASSISTANT_KNOWLEDGE_DIR`. `--only KEY ...` fetches just those papers again.
Needs pypdf (requirements-knowledge.txt), for building only.
"""

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Build the assistant's library of papers."

    def add_arguments(self, parser):
        parser.add_argument("action", choices=["build", "status"])
        parser.add_argument("--only", nargs="+", default=None, help="Keys to fetch again.")
        parser.add_argument("--no-embed", action="store_true",
                            help="Build for search by words alone.")

    def handle(self, *args, **options):
        import json

        from assistant import library
        from assistant.llm import EmbeddingClient

        if options["action"] == "status":
            built = library.index_dir() / "built.json"
            if not built.exists():
                raise CommandError(f"No library at {library.index_dir()}.")
            self.stdout.write(built.read_text())
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
        self.stdout.write(json.dumps(built, indent=2))
