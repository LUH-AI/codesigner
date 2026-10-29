"""The evals run only with `--eval`: each case once per `--eval-repeat`, and a
report of every run written to `--eval-report` at the end."""

import json
import os
import time
from collections import defaultdict

import pytest

from .harness import load_cases


def pytest_generate_tests(metafunc):
    if "case" not in metafunc.fixturenames:
        return
    config = metafunc.config
    cases = load_cases()
    wanted = config.getoption("--eval-case")
    if wanted:
        cases = [c for c in cases if any(c.id.startswith(w) or c.group.startswith(w) for w in wanted)]
    repeat = max(1, config.getoption("--eval-repeat"))
    params = [(case, n) for case in cases for n in range(repeat)]
    metafunc.parametrize("case,repeat", params,
                         ids=[f"{c.group}/{c.id}#{n}" for c, n in params])


def pytest_collection_modifyitems(config, items):
    if config.getoption("--eval"):
        return
    skip = pytest.mark.skip(reason="evals run only with --eval")
    for item in items:
        if "evals" in item.nodeid.split("/"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def eval_report(request):
    """`eval_report` collects each run's outcome and writes the report, with
    pass rates per case and per group, when the session ends."""
    from django.conf import settings

    outcomes = []
    yield outcomes
    if not outcomes:
        return
    by_case, by_group = defaultdict(list), defaultdict(list)
    for outcome in outcomes:
        by_case[outcome["case"]].append(outcome["passed"])
        by_group[outcome["group"]].append(outcome["passed"])
    report = {
        "model": settings.ASSISTANT_LLM_MODEL,
        "base_url": settings.ASSISTANT_LLM_BASE_URL,
        "extra_body": settings.ASSISTANT_LLM_EXTRA_BODY,
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "passed": sum(o["passed"] for o in outcomes), "runs": len(outcomes),
        "groups": {g: round(sum(v) / len(v), 2) for g, v in sorted(by_group.items())},
        "cases": {c: round(sum(v) / len(v), 2) for c, v in sorted(by_case.items())},
        "tokens": sum(o["tokens"] for o in outcomes),
        "runs_detail": outcomes,
    }
    path = request.config.getoption("--eval-report")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, default=str)
    terminal = request.config.pluginmanager.get_plugin("terminalreporter")
    if terminal is not None:
        terminal.write_line(f"\neval report: {os.path.abspath(path)}")
        terminal.write_line(f"{report['model']}: {report['passed']}/{report['runs']} runs passed")
        for group, rate in report["groups"].items():
            terminal.write_line(f"  {group:<20} {rate:.0%}")
