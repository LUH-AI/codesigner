# Deferred tests

These were written ahead of schedule and describe **database-backed** features:
the `Experiment`/`Run` models, the row↔snapshot adapter, the `import_ihpo`
management command, and the DB-driven experiment list/detail pages.

The trimmed plan has no database until the app first needs durable server-side
state (saving experiments / background runs). These will move into that step's
folder when it's scheduled. They currently fail (the models don't exist yet) —
run the scheduled suite with `pytest --ignore=tests/deferred` if that noise is
distracting.
