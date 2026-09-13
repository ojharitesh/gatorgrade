"""Analyze saved report history to produce deterministic check insights."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from gatorgrade.report_history import (
    CHECK_ID_KEY,
    CHECKS_KEY,
    HISTORY_REPORT_KEY,
    HISTORY_SAVED_AT_KEY,
    HISTORY_SKIP_DIFFERENT_SCOPE,
    STATUS_KEY,
)

# keys read from the check dictionaries stored inside a history report
DESCRIPTION_KEY = "description"
CHECK_NAME_KEY = "check"
PATH_KEY = "path"
FALLBACK_IDENTIFIER_PREFIX = "description::"
NAME_WITH_PATH = "{} ({})"

# aggregation rules, documented here so the output is reproducible
INSIGHTS_RATE_PRECISION = 4
INSIGHTS_MIN_TREND_OBS = 4
INSIGHTS_TREND_EPS = 0.05
INSIGHTS_MIN_RANK_OBS = 2

# trend labels attached to every emitted check summary
TREND_IMPROVING = "improving"
TREND_DECLINING = "declining"
TREND_STABLE = "stable"
TREND_INSUFFICIENT_DATA = "insufficient-data"

# diagnostic reasons produced by this module for check-level problems
INSIGHTS_SKIP_CHECK_ENTRY = "skip_check_entry"
INSIGHTS_MISSING_IDENTITY = "missing_identity"
INSIGHTS_INVALID_STATUS = "invalid_status"
INSIGHTS_DUPLICATE_CHECK = "duplicate_check"
INSIGHTS_MALFORMED_REPORT = "malformed_report"

# diagnostic kinds that say which layer of the data a diagnostic is about
DIAGNOSTIC_KIND_FILE = "file"
DIAGNOSTIC_KIND_REPORT = "report"
DIAGNOSTIC_KIND_CHECK = "check"

# mirrors HISTORY_SKIP_DIFFERENT_SCOPE in report_history; it is kept
# local so this module never depends on the history loader itself
HISTORY_SKIP_DIFFERENT_SCOPE_REASON = HISTORY_SKIP_DIFFERENT_SCOPE

# labels used when building diagnostic sources and details
REPORT_SOURCE_PREFIX = "report "
FILE_SKIPPED_DETAIL = "history file skipped by the loader"
MALFORMED_REPORT_DETAIL = "report does not contain a list of checks"
SKIP_CHECK_ENTRY_DETAIL = "check entry {} is not an object"
MISSING_IDENTITY_DETAIL = "check entry {} has no check_id or description"
INVALID_STATUS_DETAIL = (
    "check '{}' has an unreadable status and counts as a non-pass"
)
DUPLICATE_CHECK_DETAIL = (
    "check '{}' appears more than once and passes only if every entry passed"
)

# text rendering pieces
JSON_INDENT = 2
NEWLINE = "\n"
TEXT_TITLE = "GatorGrade Insights"
TEXT_SCOPE = "Scope: {}"
TEXT_REPORTS = "Reports inspected: {} of {} available"
TEXT_NO_REPORTS = "No valid history reports were found for this project."
TEXT_NO_CHECKS = "No checks were observed in the inspected reports."
TEXT_CHECKS_HEADER = "Checks ({}):"
TEXT_CHECK_NAME = "- {}"
TEXT_CHECK_ID = "  id: {}"
TEXT_CHECK_COUNTS = "  observations: {}  passes: {}  pass rate: {:.2%}"
TEXT_CHECK_LATEST = "  latest: {}  current streak: {} {}"
TEXT_CHECK_TREND = "  trend: {}"
TEXT_CHECK_TREND_DELTA = "  trend: {} (delta {:+.4f})"
TEXT_STATUS_PASS = "pass"
TEXT_STATUS_FAIL = "fail"
TEXT_STREAK_PASSING = "passing"
TEXT_STREAK_FAILING = "failing"
TEXT_BEST = "Best check: {}"
TEXT_WORST = "Worst check: {}"
TEXT_NO_RANKING = "none (no check has at least {} observations)"
TEXT_OTHER_SCOPE = "Reports from other projects ignored: {}"
TEXT_DIAGNOSTICS_HEADER = "Diagnostics ({}):"
TEXT_DIAGNOSTIC = "- {} {}: {} ({})"
TEXT_EMPTY = ""


class CheckObservation(BaseModel):
    """Record one observation of a check inside one saved report."""

    identifier: str
    name: str
    report_index: int
    source: str
    passed: bool
    status_valid: bool


class CheckSummary(BaseModel):
    """Summarize every observation of one check across the history."""

    identifier: str
    name: str
    observations: int
    passes: int
    pass_rate: float
    latest_status: bool
    current_pass_streak: int
    current_fail_streak: int
    trend: str
    trend_delta: float | None
    status_history: list[bool]


class HistoryDiagnostic(BaseModel):
    """Explain why a file, report, or check entry was skipped or flagged."""

    kind: str
    source: str
    reason: str
    detail: str


class InsightsReport(BaseModel):
    """Hold the complete deterministic insights for one project scope."""

    scope: str
    reports_inspected: int
    reports_available: int
    checks: list[CheckSummary]
    best_check: str | None
    worst_check: str | None
    diagnostics: list[HistoryDiagnostic]


def _report_source(payload: dict[str, Any], report_index: int) -> str:
    """Return a human-readable label for one chronological report."""
    saved_at = payload.get(HISTORY_SAVED_AT_KEY)
    if isinstance(saved_at, str) and saved_at:
        return saved_at
    return f"{REPORT_SOURCE_PREFIX}{report_index + 1}"


def _check_identifier(check: dict[Any, Any]) -> str | None:
    """Return the stable identity of a check entry, or None when absent."""
    check_id = check.get(CHECK_ID_KEY)
    if isinstance(check_id, str) and check_id:
        return check_id
    description = check.get(DESCRIPTION_KEY)
    if isinstance(description, str) and description:
        return f"{FALLBACK_IDENTIFIER_PREFIX}{description}"
    return None


def _check_name(check: dict[Any, Any], identifier: str) -> str:
    """Return a human-readable name for a check entry, with its path."""
    name = identifier
    for key in (DESCRIPTION_KEY, CHECK_NAME_KEY):
        value = check.get(key)
        if isinstance(value, str) and value:
            name = value
            break
    path = check.get(PATH_KEY)
    if isinstance(path, str) and path:
        return NAME_WITH_PATH.format(name, path)
    return name


def _make_diagnostic(
    kind: str,
    source: str,
    reason: str,
    detail: str,
) -> HistoryDiagnostic:
    """Create one structured diagnostic."""
    return HistoryDiagnostic(
        kind=kind,
        source=source,
        reason=reason,
        detail=detail,
    )


def _file_diagnostics(
    file_diagnostics: list[tuple[Path, str]],
) -> list[HistoryDiagnostic]:
    """Convert loader skip reasons into structured diagnostics."""
    return [
        _make_diagnostic(
            DIAGNOSTIC_KIND_FILE,
            path.name,
            reason,
            FILE_SKIPPED_DETAIL,
        )
        for path, reason in file_diagnostics
    ]


def _merge_duplicate(
    existing: CheckObservation,
    passed: bool,
    status_valid: bool,
) -> CheckObservation:
    """Merge a repeated entry so the check passes only if every entry passed."""
    return existing.model_copy(
        update={
            "passed": existing.passed and passed,
            "status_valid": existing.status_valid and status_valid,
        }
    )


def _observe_report(
    payload: dict[str, Any],
    report_index: int,
) -> tuple[list[CheckObservation], list[HistoryDiagnostic]]:
    """Convert the check entries of one report into observations."""
    source = _report_source(payload, report_index)
    report = payload.get(HISTORY_REPORT_KEY)
    checks = report.get(CHECKS_KEY) if isinstance(report, dict) else None
    if not isinstance(checks, list):
        diagnostic = _make_diagnostic(
            DIAGNOSTIC_KIND_REPORT,
            source,
            INSIGHTS_MALFORMED_REPORT,
            MALFORMED_REPORT_DETAIL,
        )
        return [], [diagnostic]
    observations: dict[str, CheckObservation] = {}
    diagnostics: list[HistoryDiagnostic] = []
    for entry_index, entry in enumerate(checks):
        if not isinstance(entry, dict):
            diagnostics.append(
                _make_diagnostic(
                    DIAGNOSTIC_KIND_CHECK,
                    source,
                    INSIGHTS_SKIP_CHECK_ENTRY,
                    SKIP_CHECK_ENTRY_DETAIL.format(entry_index),
                )
            )
            continue
        identifier = _check_identifier(entry)
        if identifier is None:
            diagnostics.append(
                _make_diagnostic(
                    DIAGNOSTIC_KIND_CHECK,
                    source,
                    INSIGHTS_MISSING_IDENTITY,
                    MISSING_IDENTITY_DETAIL.format(entry_index),
                )
            )
            continue
        name = _check_name(entry, identifier)
        status = entry.get(STATUS_KEY)
        status_valid = isinstance(status, bool)
        passed = status is True
        if not status_valid:
            diagnostics.append(
                _make_diagnostic(
                    DIAGNOSTIC_KIND_CHECK,
                    source,
                    INSIGHTS_INVALID_STATUS,
                    INVALID_STATUS_DETAIL.format(name),
                )
            )
        existing = observations.get(identifier)
        if existing is None:
            observations[identifier] = CheckObservation(
                identifier=identifier,
                name=name,
                report_index=report_index,
                source=source,
                passed=passed,
                status_valid=status_valid,
            )
            continue
        diagnostics.append(
            _make_diagnostic(
                DIAGNOSTIC_KIND_CHECK,
                source,
                INSIGHTS_DUPLICATE_CHECK,
                DUPLICATE_CHECK_DETAIL.format(name),
            )
        )
        observations[identifier] = _merge_duplicate(
            existing, passed, status_valid
        )
    return list(observations.values()), diagnostics


def _pass_rate(status_history: list[bool]) -> float:
    """Return the rounded fraction of passing statuses in a sequence."""
    return round(
        sum(status_history) / len(status_history),
        INSIGHTS_RATE_PRECISION,
    )


def _current_streaks(status_history: list[bool]) -> tuple[int, int]:
    """Return the current pass and fail streaks from the newest observation."""
    if not status_history:
        return 0, 0
    latest = status_history[-1]
    streak = 0
    for status in reversed(status_history):
        if status != latest:
            break
        streak += 1
    if latest:
        return streak, 0
    return 0, streak


def _compute_trend(status_history: list[bool]) -> tuple[str, float | None]:
    """Classify the trend by comparing older and newer observation halves."""
    if len(status_history) < INSIGHTS_MIN_TREND_OBS:
        return TREND_INSUFFICIENT_DATA, None
    split_index = len(status_history) // 2
    older_rate = _pass_rate(status_history[:split_index])
    newer_rate = _pass_rate(status_history[split_index:])
    trend_delta = round(newer_rate - older_rate, INSIGHTS_RATE_PRECISION)
    if trend_delta > INSIGHTS_TREND_EPS:
        return TREND_IMPROVING, trend_delta
    if trend_delta < -INSIGHTS_TREND_EPS:
        return TREND_DECLINING, trend_delta
    return TREND_STABLE, trend_delta


def _display_name(
    identifier: str,
    observations: list[CheckObservation],
) -> str:
    """Return the newest readable name recorded for a check."""
    for observation in reversed(observations):
        if observation.name != identifier:
            return observation.name
    return identifier


def _summarize_check(
    identifier: str,
    observations: list[CheckObservation],
) -> CheckSummary:
    """Aggregate the chronological observations of one check."""
    status_history = [observation.passed for observation in observations]
    pass_streak, fail_streak = _current_streaks(status_history)
    trend, trend_delta = _compute_trend(status_history)
    return CheckSummary(
        identifier=identifier,
        name=_display_name(identifier, observations),
        observations=len(status_history),
        passes=sum(status_history),
        pass_rate=_pass_rate(status_history),
        latest_status=status_history[-1],
        current_pass_streak=pass_streak,
        current_fail_streak=fail_streak,
        trend=trend,
        trend_delta=trend_delta,
        status_history=status_history,
    )


def _best_sort_key(summary: CheckSummary) -> tuple[float, int, int, str]:
    """Return the ordering key that puts the best check first."""
    return (
        -summary.pass_rate,
        -summary.observations,
        -summary.current_pass_streak,
        summary.identifier,
    )


def _worst_sort_key(summary: CheckSummary) -> tuple[float, int, int, str]:
    """Return the ordering key that puts the worst check first."""
    return (
        summary.pass_rate,
        -summary.observations,
        -summary.current_fail_streak,
        summary.identifier,
    )


def _rankable_checks(checks: list[CheckSummary]) -> list[CheckSummary]:
    """Return the checks with enough observations to be ranked."""
    return [
        summary
        for summary in checks
        if summary.observations >= INSIGHTS_MIN_RANK_OBS
    ]


def _select_best_check(checks: list[CheckSummary]) -> str | None:
    """Return the identifier of the best-performing check, if any."""
    candidates = _rankable_checks(checks)
    if not candidates:
        return None
    return min(candidates, key=_best_sort_key).identifier


def _select_worst_check(checks: list[CheckSummary]) -> str | None:
    """Return the identifier of the worst-performing check, if any."""
    candidates = _rankable_checks(checks)
    if not candidates:
        return None
    return min(candidates, key=_worst_sort_key).identifier


def build_insights_report(
    payloads: list[dict[str, Any]],
    *,
    reports_available: int,
    scope: str,
    file_diagnostics: list[tuple[Path, str]],
) -> InsightsReport:
    """Aggregate newest-first history payloads into an insights report."""
    diagnostics = _file_diagnostics(file_diagnostics)
    observations_by_check: dict[str, list[CheckObservation]] = {}
    for report_index, payload in enumerate(reversed(payloads)):
        observations, report_diagnostics = _observe_report(
            payload, report_index
        )
        diagnostics.extend(report_diagnostics)
        for observation in observations:
            observations_by_check.setdefault(
                observation.identifier, []
            ).append(observation)
    checks = [
        _summarize_check(identifier, observations_by_check[identifier])
        for identifier in sorted(observations_by_check)
    ]
    return InsightsReport(
        scope=scope,
        reports_inspected=len(payloads),
        reports_available=reports_available,
        checks=checks,
        best_check=_select_best_check(checks),
        worst_check=_select_worst_check(checks),
        diagnostics=diagnostics,
    )


def _status_label(status: bool) -> str:
    """Return the text label for a boolean status."""
    return TEXT_STATUS_PASS if status else TEXT_STATUS_FAIL


def _render_check(summary: CheckSummary) -> list[str]:
    """Render the text lines that describe one check summary."""
    if summary.current_pass_streak > 0:
        streak_length = summary.current_pass_streak
        streak_label = TEXT_STREAK_PASSING
    else:
        streak_length = summary.current_fail_streak
        streak_label = TEXT_STREAK_FAILING
    if summary.trend_delta is None:
        trend_line = TEXT_CHECK_TREND.format(summary.trend)
    else:
        trend_line = TEXT_CHECK_TREND_DELTA.format(
            summary.trend, summary.trend_delta
        )
    return [
        TEXT_CHECK_NAME.format(summary.name),
        TEXT_CHECK_ID.format(summary.identifier),
        TEXT_CHECK_COUNTS.format(
            summary.observations, summary.passes, summary.pass_rate
        ),
        TEXT_CHECK_LATEST.format(
            _status_label(summary.latest_status),
            streak_length,
            streak_label,
        ),
        trend_line,
    ]


def _check_name_for(report: InsightsReport, identifier: str | None) -> str:
    """Return the display name for a ranked check identifier."""
    if identifier is None:
        return TEXT_NO_RANKING.format(INSIGHTS_MIN_RANK_OBS)
    for summary in report.checks:
        if summary.identifier == identifier:
            return summary.name
    return identifier


def _render_diagnostics(report: InsightsReport) -> list[str]:
    """Render the diagnostics, keeping other-project reports quiet."""
    other_scope_count = 0
    visible: list[HistoryDiagnostic] = []
    for diagnostic in report.diagnostics:
        if diagnostic.reason == HISTORY_SKIP_DIFFERENT_SCOPE:
            other_scope_count += 1
        else:
            visible.append(diagnostic)
    lines: list[str] = []
    if other_scope_count > 0:
        lines.append(TEXT_OTHER_SCOPE.format(other_scope_count))
    if visible:
        lines.append(TEXT_DIAGNOSTICS_HEADER.format(len(visible)))
        lines.extend(
            TEXT_DIAGNOSTIC.format(
                diagnostic.kind,
                diagnostic.source,
                diagnostic.detail,
                diagnostic.reason,
            )
            for diagnostic in visible
        )
    return lines


def render_text(report: InsightsReport) -> str:
    """Render an insights report as deterministic plain text."""
    lines = [
        TEXT_TITLE,
        TEXT_SCOPE.format(report.scope),
        TEXT_REPORTS.format(
            report.reports_inspected, report.reports_available
        ),
        TEXT_EMPTY,
    ]
    if report.reports_inspected == 0:
        lines.append(TEXT_NO_REPORTS)
    elif not report.checks:
        lines.append(TEXT_NO_CHECKS)
    else:
        lines.append(TEXT_CHECKS_HEADER.format(len(report.checks)))
        for summary in report.checks:
            lines.extend(_render_check(summary))
        lines.append(TEXT_EMPTY)
        lines.append(
            TEXT_BEST.format(_check_name_for(report, report.best_check))
        )
        lines.append(
            TEXT_WORST.format(_check_name_for(report, report.worst_check))
        )
    diagnostic_lines = _render_diagnostics(report)
    if diagnostic_lines:
        lines.append(TEXT_EMPTY)
        lines.extend(diagnostic_lines)
    return NEWLINE.join(lines) + NEWLINE


def render_json(report: InsightsReport) -> str:
    """Render an insights report as deterministic JSON."""
    return report.model_dump_json(indent=JSON_INDENT)
