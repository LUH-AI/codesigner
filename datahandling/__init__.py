"""Data Handling: looking at, cleaning, reshaping and augmenting an
experiment's dataset before it is tuned on.

A skeleton. The pages exist, at ``Experiments / <name> / Data``, and say what
each section will do; nothing in them acts on a dataset yet. What there is to
build is listed once, in `sections.py`, so deciding what a section should be is
an edit there rather than in a template.

How the pieces are meant to fit, once they are built: Data Handling produces a
*recipe* — an ordered list of steps (choose the target, set a column's kind,
impute, encode, drop duplicates, oversample, ...) applied to the uploaded CSV
before it is split. The raw file is never rewritten. The recipe would travel in
the ``.ihpo`` with the rest of the experiment, and into an exported model, so
that predictions made outside Codesigner see exactly the transforms the tuned
model was scored on. Steps that only make sense on training rows —
augmentation, resampling — would apply inside each fold rather than before the
split, so they never leak into what a model is validated on.
"""
