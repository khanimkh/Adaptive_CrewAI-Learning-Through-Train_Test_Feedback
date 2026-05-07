# Adaptive CrewAI: Learning Through Train-Test-Feedback

A multi-agent CrewAI project that runs an interactive train/test feedback cycle for
support ticket analysis. Agents test their performance, incorporate user feedback
collected in the terminal, train on that feedback, then re-test to measure improvement.
Results are saved to disk and displayed in a live web UI.

---

## Features

- **4 YAML-configured agents** working sequentially on support ticket data
- **Interactive train/test cycle**: test → collect terminal feedback → train → test → kickoff
- **Terminal feedback prompt**: type multi-line feedback before training starts; press Enter on an empty line to finish
- **FastAPI backend** with endpoints for kickoff, latest results, and static file serving
- **Frontend web UI** for running kickoff and viewing saved cycle results, summary, and charts
- **Docker support** with DNS fix for OpenAI API connectivity
- **Auto-generated charts**: 6 PNG performance charts saved after each cycle

---

## Agents

| Agent | Role | Responsibility |
|---|---|---|
| `ticket_triage_agent` | Senior Support Triage Specialist | Classify tickets, identify risk hot spots |
| `action_recommender_agent` | Resolution Strategy Advisor | Recommend actions for high-impact issue types |
| `quality_auditor_agent` | Quality and Compliance Auditor | Review recommendations for quality and empathy |
| `reporting_agent` | Support Insights Reporter | Compile final management-ready report |

---

## Project Structure

```
src/support_train_test_crew/
├── main.py              # CLI entry point
├── api.py               # FastAPI backend + static file serving
├── crew.py              # Builds agents, tasks, and Crew from YAML
├── train_and_test.py    # Full train/test cycle orchestration
├── report_charts.py     # Generates PNG performance charts
├── config/
│   ├── agents.yaml      # Agent roles, goals, backstories
│   └── tasks.yaml       # Task descriptions with {terminal_feedback} injection
└── tools/
    └── custom_tool.py   # TicketStatsTool for CSV data access
frontend/
├── index.html           # Web UI
├── app.js               # Frontend logic (kickoff, results, charts)
└── style.css            # Styles
data/
└── support_tickets_data.csv   # Input dataset
reports/
├── support_train_test_summary.md          # Latest cycle Markdown summary
└── training/
    ├── support_train_test.pkl             # Saved training weights
    └── latest_cycle_results.json         # Latest cycle result payload
```

---

## Setup

### 1. Create `.env`
```
OPENAI_API_KEY=your_key_here
```

### 2. Install dependencies (local)
```bash
pip install -r requirements.txt
```

---

## Run

### Docker (recommended)
```bash
docker compose up --build -d
```
Open `http://localhost:8010`

> The `docker-compose.yml` includes `dns: [8.8.8.8, 8.8.4.4]` to ensure the container
> can reach the OpenAI API.

### CLI — Full Train/Test Cycle (terminal)
```bash
docker compose exec support-train-test-web python -m support_train_test_crew.main \
  --iterations 1 \
  --model gpt-4o-mini \
  --sample-size 5
```

The terminal will prompt:
```
feedback> Focus on recurring technical issues
feedback> Make recommendations more specific
feedback>        ← press Enter on empty line to finish
```

CrewAI then asks for per-task feedback (once per task per iteration = 4 prompts with `--iterations 1`).

#### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--iterations` | `1` | Number of train/test iterations |
| `--model` | `gpt-4o-mini` | OpenAI model used for `crew.test()` scoring |
| `--sample-size` | `5` | Number of ticket rows included in inputs |
| `--feedback` | `""` | Pre-written feedback text (shown as hint or used directly) |
| `--no-prompt-feedback` | off | Skip interactive prompt; use only `--feedback` text |

#### Non-interactive example (for scripts/CI)
```bash
docker compose exec support-train-test-web python -m support_train_test_crew.main \
  --iterations 1 --model gpt-4o-mini --sample-size 5 \
  --feedback "Focus on SLA breaches and technical issues" \
  --no-prompt-feedback
```

### Web — Kickoff Only
- Open `http://localhost:8010`
- Set **Sample Size** and click **Run Kickoff**
- Runs the 4-agent crew once without training and displays the output
- Click **Refresh Latest Results** to load the most recent saved train/test cycle

---

## Web API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/run/kickoff` | Run crew kickoff and return output |
| `GET` | `/results/latest` | Load latest saved cycle results, summary, and chart URLs |
| `GET` | `/` | Serve the frontend |
| `GET` | `/report-files/{name}` | Serve saved chart PNG files |

---

## Outputs

| File | Description |
|---|---|
| `reports/support_train_test_summary.md` | Markdown summary with cycle status, feedback, and chart links |
| `reports/training/support_train_test.pkl` | Saved CrewAI training weights |
| `reports/training/latest_cycle_results.json` | Full JSON result payload from the last CLI run |
| `reports/*.png` | 6 auto-generated performance charts |

### Generated Charts

- `issue-distribution.png` — pie chart of ticket counts by issue type
- `resolution-times.png` — avg resolution time per issue type
- `customer-satisfaction.png` — satisfaction rating per ticket
- `agent-performance-resolution.png` — avg resolution time per agent
- `agent-performance-satisfaction.png` — avg satisfaction per agent
- `agent-performance.png` — combined resolution + satisfaction per agent

---

## Acknowledgments

- **[DeepLearning.AI](https://www.deeplearning.ai/)** — for the *Multi AI Agent Systems with crewAI* short course that inspired the multi-agent architecture and train/test feedback patterns used in this project.
- **[Coursera](https://www.coursera.org/)** — for hosting the DeepLearning.AI course content and providing an accessible learning platform.
- **[CrewAI](https://www.crewai.com/)** — for the open-source multi-agent orchestration framework powering this project.
- **[OpenAI](https://openai.com/)** — for the language models driving agent reasoning and task execution.

