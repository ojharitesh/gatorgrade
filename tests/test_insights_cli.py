"""Tests for the insights and analyze commands in the GatorGrade CLI."""

import datetime
import json
import re
from io import BytesIO, StringIO, TextIOWrapper
from pathlib import Path
from typing import Any, Literal

import pytest
import typer
from click import BadParameter
from rich.console import Console
from typer.testing import CliRunner

from gatorgrade import main
from gatorgrade.insights import InsightsReport
from gatorgrade.main import (
    InsightsFormat,
    _resolve_insights_config,
    _run_insights,
    _write_insights_file,
    app,
)
from gatorgrade.report_history import (
    HISTORY_REPORT_KEY,
    get_history_scope,
    save_report_history,
)
from gatorgrade.validate import (
    validate_insights_last,
    validate_insights_output,
)

runner = CliRunner()

UTC = datetime.timezone.utc
ENCODING = "utf-8"
PROJECT_NAME = "Demo Project"
CONFIG_NAME = "gatorgrade.yml"
CONFIG_TEXT = "name: Demo Project\n---\n- check: ConfirmFileExists\n"
HISTORY_DIR_NAME = "h"
MISSING_CONFIG_NAME = "missing.yml"
OUTPUT_NAME = "insights-out.json"

INSIGHTS_COMMAND = "insights"
ANALYZE_COMMAND = "analyze"
CONFIG_FLAG = "--config"
HISTORY_DIR_FLAG = "--history-dir"
LAST_FLAG = "--last"
FORMAT_FLAG = "--format"
OUTPUT_FLAG = "--output"
HELP_FLAG = "--help"
ALIAS_HELP_TEXT = "Alias for the insights command."

CHECK_ALPHA = "check-alpha"
CHECK_BETA = "check-beta"
NAME_ALPHA = "Complete all TODOs"
NAME_BETA = "Use an if statement"
CHECKS_KEY = "checks"
CHECK_ID_KEY = "check_id"
STATUS_KEY = "status"
DESCRIPTION_KEY = "description"

HISTORY_YEAR = 2026
HISTORY_MONTH = 9
MALFORMED_FILE_NAME = "gatorgrade-report-20260930T000000.000000Z-bad.json"
MALFORMED_CONTENTS = "{not valid json"

ZERO = 0
ONE = 1
TWO = 2
THREE = 3
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
INVALID_LAST = 0
NEGATIVE_LAST = -3

TITLE_TEXT = "GatorGrade Insights"
NO_REPORTS_TEXT = "No valid history reports were found"
WROTE_TEXT = "Wrote"
MISSING_CONFIG_TEXT = "does not exist"
POSITIVE_INTEGER_TEXT = "must be a positive integer"
IS_DIRECTORY_TEXT = "is a directory"
NO_DIRECTORY_TEXT = "does not exist"
BOOM_MESSAGE = "insights must never run this"


ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
BORDER_PATTERN = re.compile("[─-╿|]")
WHITESPACE_PATTERN = re.compile(r"\s+")
SPACE = " "
EMPTY_TEXT = ""
WINDOWS_ENCODING = "cp1252"
# an empty newline setting stops the wrapper from translating a line feed
# into the platform line ending, which would otherwise make this
# comparison pass on Linux and fail on Windows
UNTRANSLATED_NEWLINE = ""
NARROW_TERMINAL_WIDTH = 40
COLOR_SYSTEM: Literal["standard"] = "standard"
CONSOLE_ATTRIBUTE = "console"
MARKUP_NAME = "[bold]Review caf\u00e9 notes[/bold] :smile:"
GREEN_TITLE = "\x1b[1;32mGatorGrade"
YELLOW_TITLE = "\x1b[1;33mInsights"
GREEN_PASS = "\x1b[1;32mpass"
RED_FAIL = "\x1b[1;31mfail"


def _display_report() -> InsightsReport:
    """Create a report with pass, fail, Unicode, and markup-like text."""
    return main.build_insights_report(
        [
            {
                HISTORY_REPORT_KEY: {
                    CHECKS_KEY: [
                        _check(CHECK_ALPHA, False, MARKUP_NAME),
                        _check(CHECK_BETA, True, NAME_BETA),
                    ]
                }
            }
        ],
        reports_available=ONE,
        scope=PROJECT_NAME,
        file_diagnostics=[],
    )


@pytest.mark.parametrize("instructor", [False, True])
def test_echo_insights_redirects_tables_to_cp1252_without_styling(
    monkeypatch: pytest.MonkeyPatch,
    instructor: bool,
) -> None:
    """Narrow redirected Windows streams preserve the exact ASCII report."""
    payload = main.render_text(_display_report(), instructor=instructor)
    buffer = BytesIO()
    with TextIOWrapper(
        buffer,
        encoding=WINDOWS_ENCODING,
        newline=UNTRANSLATED_NEWLINE,
    ) as stream:
        monkeypatch.setattr(
            main,
            CONSOLE_ATTRIBUTE,
            Console(
                file=stream,
                force_terminal=False,
                width=NARROW_TERMINAL_WIDTH,
            ),
        )
        main._echo_insights(payload)
        stream.flush()
        assert buffer.getvalue() == payload.encode(WINDOWS_ENCODING)


def test_echo_insights_colors_tables_without_changing_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal styling colors titles and statuses without interpreting names."""
    payload = main.render_text(_display_report())
    stream = StringIO()
    monkeypatch.setattr(
        main,
        CONSOLE_ATTRIBUTE,
        Console(
            file=stream,
            force_terminal=True,
            no_color=False,
            color_system=COLOR_SYSTEM,
            width=NARROW_TERMINAL_WIDTH,
        ),
    )
    main._echo_insights(payload)
    output = stream.getvalue()
    assert ANSI_PATTERN.sub(EMPTY_TEXT, output) == payload
    for colored_text in (GREEN_TITLE, YELLOW_TITLE, GREEN_PASS, RED_FAIL):
        assert colored_text in output


def test_echo_insights_keeps_json_raw_in_color_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Color support never adds ANSI codes or wrapping to JSON."""
    payload = main.render_json(_display_report())
    stream = StringIO()
    monkeypatch.setattr(
        main,
        CONSOLE_ATTRIBUTE,
        Console(
            file=stream,
            force_terminal=True,
            no_color=False,
            color_system=COLOR_SYSTEM,
            width=NARROW_TERMINAL_WIDTH,
        ),
    )
    main._echo_insights(payload)
    assert stream.getvalue().rstrip() == payload


def _plain(text: str) -> str:
    """Return CLI output without styling, borders, or wrapped line breaks."""
    without_styling = ANSI_PATTERN.sub(EMPTY_TEXT, text)
    without_borders = BORDER_PATTERN.sub(SPACE, without_styling)
    return WHITESPACE_PATTERN.sub(SPACE, without_borders)


def _write_config(tmp_path: Path) -> Path:
    """Create a configuration file carrying a project name in front matter."""
    config_path = tmp_path / CONFIG_NAME
    config_path.write_text(CONFIG_TEXT, encoding=ENCODING)
    return config_path


def _check(check_id: str, status: bool, description: str) -> dict[str, Any]:
    """Create one check entry as it appears inside a saved report."""
    return {
        CHECK_ID_KEY: check_id,
        STATUS_KEY: status,
        DESCRIPTION_KEY: description,
    }


def _seed_history(
    config_path: Path,
    history_dir: Path,
    runs: list[list[dict[str, Any]]],
) -> Path:
    """Save one history report per run, oldest first, for this config."""
    scope = get_history_scope(config_path, PROJECT_NAME)
    for day, checks in enumerate(runs, start=ONE):
        save_report_history(
            {CHECKS_KEY: checks},
            scope=scope,
            history_directory=history_dir,
            current_time=datetime.datetime(
                HISTORY_YEAR, HISTORY_MONTH, day, tzinfo=UTC
            ),
        )
    return history_dir


def _project(tmp_path: Path) -> tuple[Path, Path]:
    """Create a config plus a three-report history and return both paths."""
    config_path = _write_config(tmp_path)
    history_dir = tmp_path / HISTORY_DIR_NAME
    _seed_history(
        config_path,
        history_dir,
        [
            [
                _check(CHECK_ALPHA, False, NAME_ALPHA),
                _check(CHECK_BETA, True, NAME_BETA),
            ],
            [
                _check(CHECK_ALPHA, False, NAME_ALPHA),
                _check(CHECK_BETA, True, NAME_BETA),
            ],
            [_check(CHECK_ALPHA, True, NAME_ALPHA)],
        ],
    )
    return config_path, history_dir


def _invoke(command: str, *arguments: str) -> Any:
    """Invoke one CLI command with the given arguments."""
    return runner.invoke(app, [command, *arguments])


def test_validate_insights_last_accepts_positive_integers() -> None:
    """A positive report count passes validation unchanged."""
    assert validate_insights_last(ONE) == ONE
    assert validate_insights_last(THREE) == THREE


def test_validate_insights_last_rejects_zero_and_negative() -> None:
    """A non-positive report count raises BadParameter."""
    with pytest.raises(BadParameter):
        validate_insights_last(INVALID_LAST)
    with pytest.raises(BadParameter):
        validate_insights_last(NEGATIVE_LAST)


def test_validate_insights_last_rejects_boolean_values() -> None:
    """A boolean is not accepted even though it is an integer subclass."""
    with pytest.raises(BadParameter):
        validate_insights_last(True)


def test_validate_insights_output_allows_none_and_valid_paths(
    tmp_path: Path,
) -> None:
    """An omitted path is allowed and a writable path passes unchanged."""
    assert validate_insights_output(None) is None
    destination = tmp_path / OUTPUT_NAME
    assert validate_insights_output(destination) == destination


def test_validate_insights_output_rejects_a_directory(tmp_path: Path) -> None:
    """An existing directory is not a valid output file."""
    with pytest.raises(BadParameter):
        validate_insights_output(tmp_path)


def test_validate_insights_output_rejects_a_missing_parent(
    tmp_path: Path,
) -> None:
    """A path inside a missing directory raises BadParameter."""
    with pytest.raises(BadParameter):
        validate_insights_output(tmp_path / HISTORY_DIR_NAME / OUTPUT_NAME)


def test_resolve_insights_config_returns_an_existing_file(
    tmp_path: Path,
) -> None:
    """An existing configuration file resolves to its own path."""
    config_path = _write_config(tmp_path)
    assert _resolve_insights_config(config_path, None) == config_path


def test_resolve_insights_config_exits_when_the_file_is_missing(
    tmp_path: Path,
) -> None:
    """A missing configuration file exits instead of raising a traceback."""
    with pytest.raises(typer.Exit):
        _resolve_insights_config(tmp_path / MISSING_CONFIG_NAME, None)


def test_write_insights_file_reports_an_unwritable_destination(
    tmp_path: Path,
) -> None:
    """A destination that cannot be written exits without a traceback."""
    with pytest.raises(typer.Exit):
        _write_insights_file(tmp_path, TITLE_TEXT, InsightsFormat.TEXT)


def test_run_insights_writes_the_selected_format(tmp_path: Path) -> None:
    """The helper writes JSON to the file when JSON output is selected."""
    config_path, history_dir = _project(tmp_path)
    destination = tmp_path / OUTPUT_NAME
    _run_insights(
        config_path,
        None,
        THREE,
        InsightsFormat.JSON,
        destination,
        history_dir,
    )
    decoded = json.loads(destination.read_text(encoding=ENCODING))
    assert decoded["reports_inspected"] == THREE


def test_insights_command_renders_a_text_summary(tmp_path: Path) -> None:
    """The insights command prints a readable summary of the history."""
    config_path, history_dir = _project(tmp_path)
    result = _invoke(
        INSIGHTS_COMMAND,
        CONFIG_FLAG,
        str(config_path),
        HISTORY_DIR_FLAG,
        str(history_dir),
    )
    assert result.exit_code == EXIT_SUCCESS
    assert TITLE_TEXT in _plain(result.stdout)
    assert NAME_ALPHA in result.stdout
    assert NAME_BETA in result.stdout


def test_analyze_alias_matches_the_insights_command(tmp_path: Path) -> None:
    """The analyze alias produces exactly the same output as insights."""
    config_path, history_dir = _project(tmp_path)
    arguments = (
        CONFIG_FLAG,
        str(config_path),
        HISTORY_DIR_FLAG,
        str(history_dir),
    )
    insights_result = _invoke(INSIGHTS_COMMAND, *arguments)
    analyze_result = _invoke(ANALYZE_COMMAND, *arguments)
    assert insights_result.exit_code == EXIT_SUCCESS
    assert analyze_result.exit_code == EXIT_SUCCESS
    assert analyze_result.stdout == insights_result.stdout


def test_analyze_is_presented_as_an_alias_in_the_help() -> None:
    """The help menu lists analyze as an alias rather than a second command."""
    result = runner.invoke(app, [HELP_FLAG])
    plain_help = _plain(result.stdout)
    assert INSIGHTS_COMMAND in plain_help
    assert ANALYZE_COMMAND in plain_help
    assert ALIAS_HELP_TEXT in plain_help


def test_analyze_help_still_documents_every_option() -> None:
    """The alias exposes the same options as the insights command."""
    insights_help = _plain(_invoke(INSIGHTS_COMMAND, HELP_FLAG).stdout)
    analyze_help = _plain(_invoke(ANALYZE_COMMAND, HELP_FLAG).stdout)
    for flag in (CONFIG_FLAG, LAST_FLAG, FORMAT_FLAG, OUTPUT_FLAG):
        assert flag in insights_help
        assert flag in analyze_help


def test_insights_command_emits_valid_json(tmp_path: Path) -> None:
    """The JSON format round-trips through the insights report model."""
    config_path, history_dir = _project(tmp_path)
    result = _invoke(
        INSIGHTS_COMMAND,
        CONFIG_FLAG,
        str(config_path),
        HISTORY_DIR_FLAG,
        str(history_dir),
        FORMAT_FLAG,
        InsightsFormat.JSON.value,
    )
    assert result.exit_code == EXIT_SUCCESS
    report = InsightsReport.model_validate_json(result.stdout)
    assert report.reports_inspected == THREE
    assert report.best_check == CHECK_BETA
    assert report.worst_check == CHECK_ALPHA


def test_insights_command_never_counts_an_absent_check(
    tmp_path: Path,
) -> None:
    """A check missing from the newest report is not counted as a pass."""
    config_path, history_dir = _project(tmp_path)
    result = _invoke(
        INSIGHTS_COMMAND,
        CONFIG_FLAG,
        str(config_path),
        HISTORY_DIR_FLAG,
        str(history_dir),
        FORMAT_FLAG,
        InsightsFormat.JSON.value,
    )
    report = InsightsReport.model_validate_json(result.stdout)
    summaries = {summary.identifier: summary for summary in report.checks}
    assert summaries[CHECK_BETA].observations == TWO
    assert summaries[CHECK_ALPHA].observations == THREE


def test_insights_command_limits_the_inspected_reports(
    tmp_path: Path,
) -> None:
    """The last option restricts how many recent reports are analyzed."""
    config_path, history_dir = _project(tmp_path)
    result = _invoke(
        INSIGHTS_COMMAND,
        CONFIG_FLAG,
        str(config_path),
        HISTORY_DIR_FLAG,
        str(history_dir),
        LAST_FLAG,
        str(ONE),
        FORMAT_FLAG,
        InsightsFormat.JSON.value,
    )
    report = InsightsReport.model_validate_json(result.stdout)
    assert report.reports_inspected == ONE
    assert report.reports_available == THREE


@pytest.mark.parametrize("enable_color", [False, True])
def test_insights_command_writes_a_file_and_keeps_terminal_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enable_color: bool,
) -> None:
    """Writing a file still prints the summary and a confirmation line."""
    monkeypatch.setattr(
        main,
        CONSOLE_ATTRIBUTE,
        Console(force_terminal=enable_color, no_color=not enable_color),
    )
    config_path, history_dir = _project(tmp_path)
    destination = tmp_path / OUTPUT_NAME
    result = _invoke(
        INSIGHTS_COMMAND,
        CONFIG_FLAG,
        str(config_path),
        HISTORY_DIR_FLAG,
        str(history_dir),
        FORMAT_FLAG,
        InsightsFormat.JSON.value,
        OUTPUT_FLAG,
        str(destination),
    )
    assert result.exit_code == EXIT_SUCCESS
    assert TITLE_TEXT in _plain(result.stdout)
    assert WROTE_TEXT in result.stdout
    written = json.loads(destination.read_text(encoding=ENCODING))
    assert written["reports_inspected"] == THREE


def test_insights_command_handles_an_empty_history(tmp_path: Path) -> None:
    """An empty history reports nothing to analyze and still succeeds."""
    config_path = _write_config(tmp_path)
    history_dir = tmp_path / HISTORY_DIR_NAME
    history_dir.mkdir()
    result = _invoke(
        INSIGHTS_COMMAND,
        CONFIG_FLAG,
        str(config_path),
        HISTORY_DIR_FLAG,
        str(history_dir),
    )
    assert result.exit_code == EXIT_SUCCESS
    assert NO_REPORTS_TEXT in result.stdout


def test_insights_command_handles_a_missing_history_directory(
    tmp_path: Path,
) -> None:
    """A history directory that does not exist is not an error."""
    config_path = _write_config(tmp_path)
    result = _invoke(
        INSIGHTS_COMMAND,
        CONFIG_FLAG,
        str(config_path),
        HISTORY_DIR_FLAG,
        str(tmp_path / HISTORY_DIR_NAME),
    )
    assert result.exit_code == EXIT_SUCCESS
    assert NO_REPORTS_TEXT in result.stdout


def test_insights_command_skips_a_malformed_history_file(
    tmp_path: Path,
) -> None:
    """A malformed history file is reported rather than raising an error."""
    config_path, history_dir = _project(tmp_path)
    malformed = history_dir / MALFORMED_FILE_NAME
    malformed.write_text(MALFORMED_CONTENTS, encoding=ENCODING)
    result = _invoke(
        INSIGHTS_COMMAND,
        CONFIG_FLAG,
        str(config_path),
        HISTORY_DIR_FLAG,
        str(history_dir),
        FORMAT_FLAG,
        InsightsFormat.JSON.value,
    )
    assert result.exit_code == EXIT_SUCCESS
    report = InsightsReport.model_validate_json(result.stdout)
    assert report.reports_inspected == THREE
    assert len(report.diagnostics) == ONE


def test_insights_command_ignores_another_projects_history(
    tmp_path: Path,
) -> None:
    """Reports saved under a different scope are not analyzed."""
    config_path, history_dir = _project(tmp_path)
    other_config = tmp_path / MISSING_CONFIG_NAME
    save_report_history(
        {CHECKS_KEY: [_check(CHECK_BETA, False, NAME_BETA)]},
        scope=get_history_scope(other_config, None),
        history_directory=history_dir,
        current_time=datetime.datetime(
            HISTORY_YEAR, HISTORY_MONTH, THREE, tzinfo=UTC
        ),
    )
    result = _invoke(
        INSIGHTS_COMMAND,
        CONFIG_FLAG,
        str(config_path),
        HISTORY_DIR_FLAG,
        str(history_dir),
        FORMAT_FLAG,
        InsightsFormat.JSON.value,
    )
    report = InsightsReport.model_validate_json(result.stdout)
    assert report.reports_inspected == THREE
    assert report.reports_available == THREE


def test_insights_command_reports_a_missing_configuration(
    tmp_path: Path,
) -> None:
    """A missing configuration file fails cleanly with a useful message."""
    result = _invoke(
        INSIGHTS_COMMAND,
        CONFIG_FLAG,
        str(tmp_path / MISSING_CONFIG_NAME),
        HISTORY_DIR_FLAG,
        str(tmp_path),
    )
    assert result.exit_code == EXIT_FAILURE
    assert MISSING_CONFIG_TEXT in _plain(result.stdout)


def test_insights_command_rejects_a_non_positive_last(tmp_path: Path) -> None:
    """A zero value for the last option is rejected before any analysis."""
    config_path, history_dir = _project(tmp_path)
    result = _invoke(
        INSIGHTS_COMMAND,
        CONFIG_FLAG,
        str(config_path),
        HISTORY_DIR_FLAG,
        str(history_dir),
        LAST_FLAG,
        str(INVALID_LAST),
    )
    assert result.exit_code != EXIT_SUCCESS
    assert POSITIVE_INTEGER_TEXT in _plain(result.output)


def test_insights_command_rejects_a_directory_output(tmp_path: Path) -> None:
    """An output path that names a directory is rejected."""
    config_path, history_dir = _project(tmp_path)
    result = _invoke(
        INSIGHTS_COMMAND,
        CONFIG_FLAG,
        str(config_path),
        HISTORY_DIR_FLAG,
        str(history_dir),
        OUTPUT_FLAG,
        str(history_dir),
    )
    assert result.exit_code != EXIT_SUCCESS
    assert IS_DIRECTORY_TEXT in _plain(result.output)


def test_insights_command_rejects_a_missing_output_directory(
    tmp_path: Path,
) -> None:
    """An output path inside a missing directory is rejected."""
    config_path, history_dir = _project(tmp_path)
    result = _invoke(
        INSIGHTS_COMMAND,
        CONFIG_FLAG,
        str(config_path),
        HISTORY_DIR_FLAG,
        str(history_dir),
        OUTPUT_FLAG,
        str(tmp_path / MISSING_CONFIG_NAME / OUTPUT_NAME),
    )
    assert result.exit_code != EXIT_SUCCESS
    assert NO_DIRECTORY_TEXT in _plain(result.output)


def test_insights_command_runs_no_checks_or_configuration_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command never parses checks, runs them, or hints while analyzing."""

    def _explode(*arguments: Any, **keywords: Any) -> Any:
        """Fail loudly when a check-running code path is reached."""
        raise AssertionError(BOOM_MESSAGE)

    config_path, history_dir = _project(tmp_path)
    monkeypatch.setattr(main, "parse_config", _explode)
    monkeypatch.setattr(main, "run_checks", _explode)
    monkeypatch.setattr(main, "create_auto_hint_engine", _explode)
    result = _invoke(
        INSIGHTS_COMMAND,
        CONFIG_FLAG,
        str(config_path),
        HISTORY_DIR_FLAG,
        str(history_dir),
    )
    assert result.exit_code == EXIT_SUCCESS
    assert TITLE_TEXT in _plain(result.stdout)
