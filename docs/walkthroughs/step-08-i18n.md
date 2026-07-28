# Step 8 — Use the app in another language

**Status:** implemented, awaiting your sign-off.
**You can now:** switch the interface between English, German, and Spanish from a dropdown in the sidebar. The choice is remembered across pages and restarts (a cookie), and every visible string — sidebar, forms, run controls, result panels, dialogs — comes back translated.

---

## 1. The Django concepts this introduces

Django's i18n is `gettext` wired into the framework. The moving parts:

- **`USE_I18N` + `LocaleMiddleware`.** The middleware sits between
  `SessionMiddleware` and `CommonMiddleware`; on every request it decides the
  active language (from the `django_language` cookie, then the
  `Accept-Language` header, then `LANGUAGE_CODE`) and activates it for the
  duration of that request.
- **`LANGUAGES` + `LANGUAGE_CODE` + `LOCALE_PATHS`.** `LANGUAGES` is the menu of
  offered locales; `LANGUAGE_CODE` is the default (`en`); `LOCALE_PATHS` tells
  Django where the catalogs live (`locale/`).
- **Marking strings.** In templates, `{% load i18n %}` then `{% translate "…" %}`
  for a plain string and `{% blocktranslate %}…{{ var }}…{% endblocktranslate %}`
  when a variable is interpolated. In Python, `gettext` (immediate — used in
  views, where a request is in flight) and `gettext_lazy` (deferred — used for
  form field labels and other module-level strings that are defined once at
  import but must translate per-request).
- **The catalogs.** `makemessages` scans the marked strings and writes a
  `django.po` per locale (msgid = the English source, msgstr = the translation);
  `compilemessages` turns each `.po` into the binary `.mo` that gettext actually
  reads at runtime.
- **`set_language`.** Django's built-in view (mounted at `/i18n/`), which the
  sidebar dropdown POSTs to; it writes the language cookie and redirects back.

The key reason this was cheap: the Streamlit app already keyed its translations
on the **English text** (`utils/strings.py` maps a symbolic key → English, and
the `.po` msgids are those English strings). Django's `{% translate %}` also
keys on the English source. So a translation that is correct in Streamlit is
correct here **verbatim** — the msgid is identical.

## 2. What was ported

Source: `utils/strings.py` (the string registry), `locale/{de,es}/LC_MESSAGES/app.po`
(the human-verified German/Spanish translations), and `utils/check_translations.py`
(the completeness check).

- **The translations themselves.** Of the 70 strings Codesigner marks, **30 are
  shared verbatim** with the Streamlit catalog (`Experiments`, `Number of
  trials`, `▶ Run`, `Best configuration`, …) and are reused byte-for-byte. The
  other 40 are Codesigner-native — either strings the Streamlit app phrases
  differently, or strings that only exist here (the standalone Cancel-run
  button, the read-only-dataset prompt, the trials table header). Where a native
  string was the Streamlit one with only a placeholder syntax difference
  (`{trial}` → Django's `%(trial)s`), its translation was adapted from the
  oracle; the genuinely new ones were translated fresh, in the same register.
- **The completeness check** (`utils/check_translations.py`) is reproduced as a
  test (below) rather than a standalone script — it fits the project's
  test-first discipline and runs in CI with everything else.

## 3. Feature checklist

✅ done · 🔄 changed.

- ✅ Three languages offered: English, Deutsch, Español
- ✅ Language dropdown in the sidebar, posting to `set_language`, returning to
  the current page
- ✅ Choice persists across pages and restarts (cookie, not session state)
- ✅ Every template string marked (`base.html`, `home.html`,
  `experiment_detail.html` + its partials `_selected_config_inner.html` /
  `_run_status.html`, `new_experiment.html`, `import.html`, `inspect.html`,
  `metric_change.html`, `delete_confirm.html`)
- ✅ Form field labels + validation errors marked (`forms.py`, via
  `gettext_lazy`)
- ✅ User-facing view messages marked (`views.py`: the *Inconsistent* metric
  label, the import errors)
- ✅ German and Spanish catalogs complete — every marked string translated
  (the CI check enforces this)
- 🔄 The `utils/strings.py` symbolic-key shim is **not** carried over — Django
  keys directly on the English source, so the indirection isn't needed (this
  was a plan decision, restated here for the record).

## 4. What is *not* translated (by design)

Data, not chrome: dataset/model/optimizer names, metric names, hyperparameter
names and their values, and trial numbers are content, not UI labels, and stay
as-is in every language. The `Run` model's status label inside the "Optimizing…
(Running)" line is left in English — it's an internal state name, and the
surrounding sentence is translated.

## 5. The tests (`tests/ui/i18n/`)

Written before the catalogs existed (they failed red until the `.po`/`.mo`
files were built), then made green:

- **`test_every_marked_string_is_translated`** (de, es) — the completeness
  check, adapted from `utils/check_translations.py`: parses each `django.po` and
  asserts no marked msgid has an empty translation. This is the CI guard the
  plan asked for.
- **`test_translations_match_interactivehpo_reference`** (de, es) — the parity
  link to the oracle: for every msgid Codesigner *shares* with the Streamlit
  app, our translation must be byte-identical to theirs. Not a hand-copied
  expectation table — it reads both catalogs and diffs them, so drift on either
  side is caught. Skips cleanly if the sibling InteractiveHPO checkout isn't
  present.
- **`test_gettext_translates_a_known_string_to_{german,spanish}`** — the gettext
  machinery is actually wired (LOCALE_PATHS + compiled `.mo`): under an
  activated locale, `gettext("Number of trials")` returns the translation.
- **`test_default_language_renders_english`** / **`…_switcher_lists_all_three`**
  / **`test_set_language_switches_rendered_page`** (de, es) — the runtime flow:
  default renders English, the switcher offers all three, and POSTing to
  `set_language` makes the next render come back translated.

## 6. Verify it yourself

```bash
cd ~/Downloads/python/projects/codesigner
.venv/bin/python manage.py compilemessages -l de -l es   # build the .mo files
.venv/bin/python -m pytest tests/ui/i18n                   # 10 tests
.venv/bin/python manage.py runserver
```

Pick **Deutsch** or **Español** from the 🌐 dropdown top-left — the whole UI
switches and stays switched as you navigate. Compare any page against the same
page in the Streamlit app (`streamlit run app/main.py`, language selector in its
sidebar): the translated labels match.

I also verified the cookie flow live (no headless browser here, so via HTTP):
started the server, POSTed `language=de` then `language=es` to `/i18n/setlang/`,
and confirmed the home page came back with `Experimente` / `Erstellen Sie über
die Seitenleiste …` and then `Experimentos` / `Usa la barra lateral …`
respectively. Full suite: 169 tests pass.

> **Note on numbering.** The approved PLAN calls this Step 8; the original
> greenfield plan listed i18n as Step 9. Same work — the plan was renumbered
> when the interim steps were consolidated.
