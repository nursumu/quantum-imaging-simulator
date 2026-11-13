"""Helpers shared by the validation suites."""
from __future__ import annotations


def check(name, passed, value, criterion, details="", category="numerical"):
    return dict(test=name, passed=bool(passed), value=value, criterion=criterion, details=details, category=category)


def assert_all(results):
    failed = [f"{r['test']}: {r['value']}" for r in results if not r["passed"]]
    assert not failed, "\n".join(failed)
