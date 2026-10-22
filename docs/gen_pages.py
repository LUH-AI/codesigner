"""The site's pages, made at build time: the README as the front page, and a
reference page for every module, read from its docstrings."""

from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).parent.parent

#: Each package, and the directory its import path starts from.
PACKAGES = {"core": ROOT, "experiments": ROOT, "access": ROOT, "ui": ROOT,
            "remote": ROOT, "config": ROOT, "codesigner_model": ROOT / "model_sdk"}
SKIPPED = {"migrations", "__pycache__"}

with mkdocs_gen_files.open("index.md", "w") as page:
    page.write((ROOT / "README.md").read_text(encoding="utf-8"))

nav = mkdocs_gen_files.Nav()
for package, base in PACKAGES.items():
    for path in sorted((base / package).rglob("*.py")):
        parts = list(path.relative_to(base).with_suffix("").parts)
        if SKIPPED & set(parts) or parts[-1] == "__main__":
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc = Path(*parts, "index.md")
        else:
            doc = Path(*parts).with_suffix(".md")
        nav[parts] = doc.as_posix()
        with mkdocs_gen_files.open(Path("reference", doc), "w") as page:
            page.write(f"::: {'.'.join(parts)}\n")
        mkdocs_gen_files.set_edit_path(Path("reference", doc), path.relative_to(ROOT))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as page:
    page.writelines(nav.build_literate_nav())
