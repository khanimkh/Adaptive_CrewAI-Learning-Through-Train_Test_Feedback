from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .train_and_test import load_latest_cycle_results, run_kickoff_only

# Strip ANSI color/control sequences so web output stays readable.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\([A-Za-z]")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
REPORT_DIR = PROJECT_ROOT / "reports"

app = FastAPI(title="Adaptive CrewAI: Learning Through Train-Test-Feedback", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CycleRequest(BaseModel):
    sample_size: int = Field(default=5, ge=1, le=100)


@app.get("/health")
def health() -> dict[str, str]:
    """Simple liveness check for container and API health probes."""
    return {"status": "ok"}


@app.post("/run/kickoff")
def run_kickoff(payload: CycleRequest) -> dict:
    """Run kickoff-only flow and return cleaned process log + final report."""
    stdout_capture = io.StringIO()
    with redirect_stdout(stdout_capture):
        result = run_kickoff_only(sample_size=payload.sample_size)

    return {
        "title": "Adaptive CrewAI: Learning Through Train-Test-Feedback",
        "process_log": _ANSI_RE.sub("", stdout_capture.getvalue().strip()),
        "summary": None,
        "final_report": result.get("final_output", ""),
        "kickoff_status": result.get("kickoff_status", "kickoff: skipped"),
    }


@app.get("/results/latest")
def latest_results() -> dict:
    """Load latest saved train/test artifacts and map chart file names to URLs."""
    result = load_latest_cycle_results()
    result["chart_urls"] = [f"/report-files/{name}" for name in result.get("chart_files", [])]
    return result


@app.get("/")
def home() -> FileResponse:
    """Serve the frontend entry page."""
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
app.mount("/report-files", StaticFiles(directory=REPORT_DIR), name="report-files")


def main() -> None:
    """Run the FastAPI app with uvicorn for local/dev execution."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8010)
