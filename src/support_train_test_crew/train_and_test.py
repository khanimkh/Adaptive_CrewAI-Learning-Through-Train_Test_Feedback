from __future__ import annotations

import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path
import sys
from typing import Any

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\([A-Za-z]")


def _strip_ansi(text: str) -> str:
    """
    Remove ANSI terminal color and control sequences from a string.

    Parameters
    ----------
    text : str
        Raw string that may contain ANSI escape codes (e.g. from CrewAI verbose output).

    Returns
    -------
    str
        The same string with all ANSI escape codes removed, safe for web display.
    """
    return _ANSI_RE.sub("", text)

from .crew import create_support_train_test_crew, default_inputs
from .report_charts import generate_report_charts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports"
TRAINING_DIR = REPORT_DIR / "training"
SUMMARY_FILE = REPORT_DIR / "support_train_test_summary.md"
TRAINED_FILE = TRAINING_DIR / "support_train_test.pkl"
LATEST_RESULTS_FILE = TRAINING_DIR / "latest_cycle_results.json"


def _safe_call(fn_name: str, fn, *args, **kwargs) -> str:
    """
    Call a function and return a status string indicating success or failure.

    Parameters
    ----------
    fn_name : str
        Human-readable label used in the returned status string.
    fn : callable
        The function to call.
    *args, **kwargs
        Positional and keyword arguments forwarded to fn.

    Returns
    -------
    str
        "<fn_name>: success" if fn completes without error,
        "<fn_name>: failed (<exception>)" otherwise.
    """
    try:
        fn(*args, **kwargs)
        return f"{fn_name}: success"
    except Exception as exc:  # noqa: BLE001
        return f"{fn_name}: failed ({exc})"


class _TeeWriter:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _call_with_log(fn_name: str, fn, *args, **kwargs) -> dict[str, Any]:
    """
    Call a function while mirroring its stdout to the terminal and capturing it.

    Stdout is tee-written so interactive prompts (e.g. CrewAI training feedback)
    remain visible in the terminal while also being stored for later use.
    ANSI color codes are stripped from the captured log before returning.

    Parameters
    ----------
    fn_name : str
        Label used in log messages and the returned status string.
    fn : callable
        The function to execute (e.g. crew.train, crew.test).
    *args, **kwargs
        Positional and keyword arguments forwarded to fn.

    Returns
    -------
    dict[str, Any]
        A dictionary with keys:
        - "status" : "<fn_name>: success" or "<fn_name>: failed (<error>)"
        - "ok"     : True if successful, False otherwise
        - "log"    : captured stdout text (ANSI stripped)
        - "error"  : exception message string, empty on success
    """
    stream = io.StringIO()
    tee = _TeeWriter(sys.stdout, stream)
    try:
        print(f"[cycle] starting: {fn_name}")
        # Mirror stdout to terminal and buffer so interactive prompts remain visible.
        with redirect_stdout(tee):
            fn(*args, **kwargs)
        captured = _strip_ansi(stream.getvalue().strip())
        print(f"[cycle] finished: {fn_name} (success)")
        return {
            "status": f"{fn_name}: success",
            "ok": True,
            "log": captured,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        captured = _strip_ansi(stream.getvalue().strip())
        print(f"[cycle] finished: {fn_name} (failed: {exc})")
        return {
            "status": f"{fn_name}: failed ({exc})",
            "ok": False,
            "log": captured,
            "error": str(exc),
        }


def _write_latest_results(results: dict[str, Any]) -> None:
    """
    Persist the cycle result payload to disk as a JSON file.

    Parameters
    ----------
    results : dict[str, Any]
        The full result dictionary from run_train_test_cycle() to save.
        Written to LATEST_RESULTS_FILE (reports/training/latest_cycle_results.json).
    """
    LATEST_RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")


def _read_latest_results() -> dict[str, Any]:
    """
    Read and parse the latest cycle results JSON file from disk.

    Returns
    -------
    dict[str, Any]
        If the file exists and is valid:
            {"available": True, "message": str, "results": dict}
        If the file does not exist or cannot be parsed:
            {"available": False, "message": str, "results": None}
    """
    if not LATEST_RESULTS_FILE.exists():
        return {
            "available": False,
            "message": "No terminal train/test run found yet.",
            "results": None,
        }

    try:
        payload = json.loads(LATEST_RESULTS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "message": f"Latest results file is unreadable: {exc}",
            "results": None,
        }

    return {
        "available": True,
        "message": "Latest terminal train/test results loaded.",
        "results": payload,
    }


def _build_summary(
    *,
    iterations: int,
    model: str,
    sample_size: int,
    user_feedback: str,
    before_test: str,
    train_status: str,
    after_test: str,
    kickoff_status: str,
    kickoff_output: str,
    charts: list[str],
) -> None:
    """
    Build and write the Markdown summary report for a completed train/test cycle.

    Parameters (all keyword-only)
    ------------------------------
    iterations : int
        Number of iterations used in the cycle.
    model : str
        OpenAI model name used for crew.test() evaluation.
    sample_size : int
        Number of ticket rows that were analysed.
    user_feedback : str
        Terminal feedback text provided by the user before training.
    before_test : str
        Status string from the pre-training test run.
    train_status : str
        Status string from the training step.
    after_test : str
        Status string from the post-training test run.
    kickoff_status : str
        Status string from the final crew kickoff.
    kickoff_output : str
        Full text output returned by crew.kickoff().
    charts : list[str]
        List of chart file names (or error strings) from generate_report_charts().

    Writes
    ------
    SUMMARY_FILE (reports/support_train_test_summary.md)
    """
    summary = [
        "# Adaptive CrewAI: Learning Through Train-Test-Feedback",
        "",
        f"- Iterations: {iterations}",
        f"- Model: {model}",
        f"- Sample Size: {sample_size}",
        f"- User Feedback (Terminal): {user_feedback or 'none'}",
        "",
        "## Cycle Results",
        f"- {before_test}",
        f"- {train_status}",
        f"- {after_test}",
        f"- {kickoff_status}",
        "",
        "## Final Crew Output",
        kickoff_output,
        "",
        "## Generated Diagrams",
    ]
    for chart in charts:
        if chart.endswith(".png"):
            summary.append(f"- {chart}")
            summary.append(f"![{chart}]({chart})")
        else:
            summary.append(f"- {chart}")
    summary.append("")

    SUMMARY_FILE.write_text("\n".join(summary), encoding="utf-8")


def _collect_terminal_feedback(default_feedback: str = "") -> str:
    """
    Prompt the user to enter multi-line training feedback in the terminal.

    Reads lines from stdin until the user submits an empty line. If no lines
    are entered, falls back to the default_feedback value.

    Parameters
    ----------
    default_feedback : str, optional
        Pre-existing feedback text shown as a hint and used as fallback
        if the user submits nothing. Defaults to "".

    Returns
    -------
    str
        The collected feedback text joined with newlines, or default_feedback
        if the user pressed Enter without typing anything.
    """
    print("[cycle] Please enter training feedback for the model.")
    print("[cycle] Press Enter on an empty line to finish.")
    if default_feedback:
        print(f"[cycle] Current default feedback: {default_feedback}")

    lines: list[str] = []
    while True:
        line = input("feedback> ").strip()
        if not line:
            break
        lines.append(line)

    if lines:
        return "\n".join(lines)
    return default_feedback.strip()


def run_train_test_cycle(
    iterations: int = 1,
    model: str = "gpt-4o-mini",
    sample_size: int = 5,
    terminal_feedback: str | None = None,
    prompt_feedback: bool = True,
) -> dict[str, Any]:
    """
    Run the full train/test cycle: test → collect feedback → train → test → kickoff.

    Parameters
    ----------
    iterations : int, optional
        Number of train/test iterations. Each iteration runs all 4 tasks once
        and prompts for per-task feedback. Defaults to 1.
    model : str, optional
        OpenAI model name used for crew.test() scoring. Defaults to "gpt-4o-mini".
    sample_size : int, optional
        Number of support ticket rows included in the crew inputs. Defaults to 5.
    terminal_feedback : str | None, optional
        Pre-written feedback text to seed the training input. If prompt_feedback
        is True, this is shown as a hint before the interactive prompt.
        Defaults to None.
    prompt_feedback : bool, optional
        If True (default), the terminal will interactively prompt the user for
        feedback before training starts. If False, only terminal_feedback is used.

    Returns
    -------
    dict[str, Any]
        Result payload with keys:
        - before_test_status   : status string from pre-training test
        - train_status         : status string from training step
        - after_test_status    : status string from post-training test
        - kickoff_status       : status string from final kickoff
        - user_feedback        : feedback text used during training
        - report_path          : path to the written Markdown summary file
        - training_file_path   : path to the saved .pkl training file
        - charts               : list of generated chart file names
        - final_output         : full text output from crew.kickoff()
        - before_test_output   : captured stdout from pre-training test
        - after_test_output    : captured stdout from post-training test
        - train_output         : captured stdout from training step
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[cycle] run started | iterations={iterations} model={model} sample_size={sample_size}")

    inputs = default_inputs(sample_size=sample_size)
    seed_feedback = (terminal_feedback or "").strip()
    user_feedback = seed_feedback
    if prompt_feedback:
        user_feedback = _collect_terminal_feedback(default_feedback=seed_feedback)
    inputs["terminal_feedback"] = user_feedback or "No explicit feedback provided."

    crew = create_support_train_test_crew(inputs)

    before_eval = _call_with_log("test_before_training", crew.test, n_iterations=iterations, openai_model_name=model)
    train_eval = _call_with_log(
        "train",
        crew.train,
        n_iterations=iterations,
        filename=str(TRAINED_FILE),
        inputs=inputs,
    )
    after_eval = _call_with_log("test_after_training", crew.test, n_iterations=iterations, openai_model_name=model)

    before_test = before_eval["status"]
    train_status = train_eval["status"]
    after_test = after_eval["status"]

    kickoff_output = ""
    kickoff_status = "kickoff: skipped"
    try:
        print("[cycle] starting: kickoff")
        result = crew.kickoff(inputs=inputs)
        kickoff_output = _strip_ansi(str(result))
        kickoff_status = "kickoff: success"
        print("[cycle] finished: kickoff (success)")
    except Exception as exc:  # noqa: BLE001
        kickoff_output = f"Kickoff failed: {exc}"
        kickoff_status = f"kickoff: failed ({exc})"
        print(f"[cycle] finished: kickoff (failed: {exc})")

    charts: list[str] = []
    try:
        charts = generate_report_charts(
            report_dir=REPORT_DIR,
            data_path=PROJECT_ROOT / "data" / "support_tickets_data.csv",
        )
    except Exception as exc:  # noqa: BLE001
        charts = [f"chart_generation_failed: {exc}"]

    result_payload = {
        "before_test_status": before_test,
        "train_status": train_status,
        "after_test_status": after_test,
        "kickoff_status": kickoff_status,
        "user_feedback": user_feedback,
        "report_path": str(SUMMARY_FILE),
        "training_file_path": str(TRAINED_FILE),
        "charts": charts,
        "final_output": kickoff_output,
        "before_test_output": before_eval.get("log", ""),
        "after_test_output": after_eval.get("log", ""),
        "train_output": train_eval.get("log", ""),
    }

    _build_summary(
        iterations=iterations,
        model=model,
        sample_size=sample_size,
        user_feedback=user_feedback,
        before_test=before_test,
        train_status=train_status,
        after_test=after_test,
        kickoff_status=kickoff_status,
        kickoff_output=kickoff_output,
        charts=charts,
    )
    _write_latest_results(result_payload)
    print("[cycle] artifacts saved: summary, latest results json, charts")

    return result_payload


def run_kickoff_only(sample_size: int = 5) -> dict[str, Any]:
    """
    Run a single crew kickoff without training or evaluation steps.

    Used by the web API to let users trigger a live crew run from the browser.
    Does not save a summary or training file — returns results directly.

    Parameters
    ----------
    sample_size : int, optional
        Number of support ticket rows included in the crew inputs. Defaults to 5.

    Returns
    -------
    dict[str, Any]
        A dictionary with keys:
        - "kickoff_status" : "kickoff: success" or "kickoff: failed (<error>)"
        - "final_output"   : full ANSI-stripped text output from crew.kickoff()
    """
    inputs = default_inputs(sample_size=sample_size)
    inputs["terminal_feedback"] = "No explicit feedback provided."
    crew = create_support_train_test_crew(inputs)

    kickoff_output = ""
    kickoff_status = "kickoff: skipped"
    try:
        result = crew.kickoff(inputs=inputs)
        kickoff_output = _strip_ansi(str(result))
        kickoff_status = "kickoff: success"
    except Exception as exc:  # noqa: BLE001
        kickoff_output = f"Kickoff failed: {exc}"
        kickoff_status = f"kickoff: failed ({exc})"

    return {
        "kickoff_status": kickoff_status,
        "final_output": kickoff_output,
    }


def load_latest_cycle_results() -> dict[str, Any]:
    """
    Load the latest saved train/test cycle artifacts for the web results page.

    Reads the latest_cycle_results.json file, the Markdown summary, and scans
    the report directory for any PNG chart files.

    Returns
    -------
    dict[str, Any]
        Merged dictionary containing:
        - "available"     : bool — whether a saved results file exists
        - "message"       : status message string
        - "results"       : raw payload dict from the last run (or None)
        - "summary_text"  : full text of the Markdown summary file (or "")
        - "chart_files"   : sorted list of PNG file names in the reports directory
        - "summary_file"  : absolute path to the Markdown summary file
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = _read_latest_results()

    summary_text = ""
    if SUMMARY_FILE.exists():
        summary_text = SUMMARY_FILE.read_text(encoding="utf-8")

    chart_files = [p.name for p in sorted(REPORT_DIR.glob("*.png"))]

    return {
        **payload,
        "summary_text": summary_text,
        "chart_files": chart_files,
        "summary_file": str(SUMMARY_FILE),
    }
