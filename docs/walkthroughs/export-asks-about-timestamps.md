# Exporting asks, rather than remembering

## What changed

`export_absolute_times` was a checkbox on the experiment settings page. It is
gone. Following the Export link now opens a page with three ways out: **Export
without times**, **Export with times**, and Cancel.

## Why a question rather than a setting

A setting is answered once, by whoever set the instance up, for every file it
will ever send. The right answer here depends on the file and on who is about to
receive it — neither of which is known while a settings page is being filled in.
It also has to be found and understood *before* the question has been raised,
which is the wrong order for a question about what you are handing over.

What is at stake is worth stating rather than naming. `starttime` and `endtime`
say what time of day someone was working and on which days: a fact about a
person, not about a search. So the page says that, and says what leaving them out
costs — nothing else. Every score, every configuration and every trial's duration
are in the file either way, which is why the safer answer can be the default one.

Two buttons rather than a button and a checkbox: a checkbox has a state to
notice, and this is worth a decision each time.

## Shape

`experiment_export` renders `ui/export_confirm.html` on GET and streams the file
on POST, reading `timestamps=keep|strip`. Anything that is not `keep` strips —
so a stale bookmark or a script written against the old URL gets the file
without them rather than the file with.

`ui/_confirm_page.html` already had the shape: a heading, the consequence
stated plainly, the buttons that carry it out, and Cancel going back where the
reader came from. This is its fourth user and the first with two ways to say yes.

Tests reach the file through `export_ihpo` in `tests/conftest.py`, which posts
the answer — worth a helper rather than a POST spelled out in eight files.

## Verify

```
python -m pytest -q -m "not slow"     # 990 passing
python manage.py check
python manage.py makemigrations --check --dry-run
```

`tests/ui/storage/test_export_timestamps.py` (moved from
`tests/ui/settings/test_export_scrub.py`, since it is no longer about a setting)
covers the page coming before the file, both answers, the default for a request
that says nothing, and that a stripped file still re-imports.
