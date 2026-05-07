from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, Process, Task

from .tools import TicketStatsTool


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    """
    Read a YAML configuration file from disk and parse it.

    Parameters
    ----------
    path : Path
        Absolute or relative path to the YAML file to load.

    Returns
    -------
    dict[str, Any]
        Parsed contents of the YAML file as a Python dictionary.
    """
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_support_train_test_crew(inputs: dict[str, Any] | None = None) -> Crew:
    """
    Build and return a fully configured CrewAI Crew for support ticket analysis.

    Reads agent definitions from config/agents.yaml and task definitions from
    config/tasks.yaml. Assigns the TicketStatsTool to the ticket_triage_agent.
    Tasks are wired together in dependency order using the context field in the
    YAML so each agent receives relevant outputs from upstream agents.

    Parameters
    ----------
    inputs : dict[str, Any] | None
        Optional runtime inputs forwarded to tasks. Expected keys:
        - "project_name"   : display name of the project (str)
        - "sample_size"    : number of tickets to analyse (int)
        - "dataset_path"   : absolute path to the CSV data file (str)
        - "terminal_feedback" : user feedback text injected into task prompts (str)
        If None, an empty dict is used and defaults apply.

    Returns
    -------
    Crew
        A CrewAI Crew instance ready for kickoff(), train(), or test() calls.
    """
    inputs = inputs or {}
    agent_cfg = _load_yaml(CONFIG_DIR / "agents.yaml")
    task_cfg = _load_yaml(CONFIG_DIR / "tasks.yaml")

    ticket_tool = TicketStatsTool()

    agents: dict[str, Agent] = {}
    for name, cfg in agent_cfg.items():
        tools = [ticket_tool] if name == "ticket_triage_agent" else []
        agents[name] = Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            verbose=cfg.get("verbose", True),
            tools=tools,
        )

    tasks: list[Task] = []
    task_lookup: dict[str, Task] = {}
    for name, cfg in task_cfg.items():
        # Resolve declared task dependencies so upstream outputs feed downstream tasks.
        task_context = [task_lookup[key] for key in cfg.get("context", []) if key in task_lookup]
        task = Task(
            description=cfg["description"],
            expected_output=cfg["expected_output"],
            agent=agents[cfg["agent"]],
            context=task_context,
        )
        tasks.append(task)
        task_lookup[name] = task

    return Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        memory=False,
    )


def default_inputs(sample_size: int = 5) -> dict[str, Any]:
    """
    Build the standard runtime inputs dictionary used across all cycle operations.

    Parameters
    ----------
    sample_size : int, optional
        Number of support ticket rows to include in the analysis sample.
        Defaults to 5. Passed through to agent task prompts via {sample_size}.

    Returns
    -------
    dict[str, Any]
        A dictionary with the following keys:
        - "project_name"   : human-readable project label used in reports
        - "sample_size"    : number of tickets to analyse
        - "dataset_path"   : absolute path to support_tickets_data.csv
    """
    csv_path = PROJECT_ROOT / "data" / "support_tickets_data.csv"
    return {
        "project_name": "Adaptive CrewAI: Learning Through Train-Test-Feedback",
        "sample_size": sample_size,
        "dataset_path": str(csv_path),
    }
