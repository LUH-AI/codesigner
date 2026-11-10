"""Library builds take minutes."""

from huey.contrib.djhuey import db_task


@db_task(priority=5)
def build_library_task():
    """`build_library_task` skips fresh libraries."""
    import logging

    from .library import build, stale
    from .llm import EmbeddingClient

    log = logging.getLogger(__name__)
    embedder = EmbeddingClient.from_settings()
    reason = stale(getattr(embedder, "model", None))
    if not reason:
        return None
    log.info("Building the assistant's library: %s.", reason)
    return build(embedder=embedder, progress=log.info)
