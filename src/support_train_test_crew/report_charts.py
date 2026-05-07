from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_CHART_NAMES = [
    "issue-distribution.png",
    "resolution-times.png",
    "customer-satisfaction.png",
    "agent-performance-resolution.png",
    "agent-performance-satisfaction.png",
    "agent-performance.png",
    "status-summary.png",
]


def _load_dataset(data_path: Path) -> pd.DataFrame:
    """
    Load the support tickets CSV file and validate required columns exist.

    Parameters
    ----------
    data_path : Path
        Absolute path to the CSV file containing support ticket records.

    Returns
    -------
    pd.DataFrame
        DataFrame with at minimum these columns:
        ticket_id, issue_type, priority, resolution_time_minutes,
        satisfaction_rating.

    Raises
    ------
    ValueError
        If any required columns are missing from the CSV file.
    """
    df = pd.read_csv(data_path)
    required = {
        "ticket_id",
        "issue_type",
        "priority",
        "resolution_time_minutes",
        "satisfaction_rating",
    }
    missing = required - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
    return df


def _build_agent_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-agent summary table with resolution and satisfaction metrics.

    Since the raw dataset may not contain agent IDs, synthetic IDs (A001-A005)
    are assigned by cycling through ticket rows before aggregation.

    Parameters
    ----------
    df : pd.DataFrame
        Full support tickets DataFrame as returned by _load_dataset().

    Returns
    -------
    pd.DataFrame
        One row per agent with columns:
        - agent_id         : synthetic agent identifier (e.g. "A001")
        - avg_resolution   : mean resolution time in minutes
        - avg_satisfaction : mean customer satisfaction rating
        - total_tickets    : total number of tickets handled
    """
    # Build synthetic agent ownership for charting when raw agent ids are unavailable.
    work = df.copy()
    work["agent_id"] = [f"A{(i % 5) + 1:03d}" for i in range(len(work))]
    grouped = (
        work.groupby("agent_id", as_index=False)
        .agg(
            avg_resolution=("resolution_time_minutes", "mean"),
            avg_satisfaction=("satisfaction_rating", "mean"),
            total_tickets=("ticket_id", "count"),
        )
        .sort_values("agent_id")
    )
    return grouped
def generate_report_charts(report_dir: Path, data_path: Path) -> list[str]:
    """
    Generate all report charts from the support tickets dataset and save them as PNG files.

    Produces the following charts:
    - issue-distribution.png              : pie chart of ticket counts by issue type
    - resolution-times.png                : bar chart of avg resolution time per issue type
    - customer-satisfaction.png           : line chart of satisfaction rating per ticket
    - agent-performance-resolution.png    : bar chart of avg resolution time per agent
    - agent-performance-satisfaction.png  : bar chart of avg satisfaction per agent
    - agent-performance.png               : combined bar+line chart for both metrics per agent

    Parameters
    ----------
    report_dir : Path
        Directory where chart PNG files will be written. Created if it does not exist.
    data_path : Path
        Absolute path to the support tickets CSV file used as chart data source.

    Returns
    -------
    list[str]
        Ordered list of PNG file names (not full paths) that were successfully written.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = _load_dataset(data_path)

    issue_counts = df["issue_type"].value_counts().sort_values(ascending=False)
    issue_resolution = (
        df.groupby("issue_type", as_index=False)["resolution_time_minutes"]
        .mean()
        .sort_values("resolution_time_minutes", ascending=False)
    )
    customer_curve = df.sort_values("ticket_id")
    agent_df = _build_agent_metrics(df)

    written: list[str] = []

    plt.figure(figsize=(10, 6))
    plt.pie(issue_counts.values, labels=issue_counts.index, autopct="%1.1f%%", startangle=140)
    plt.title("Issue Distribution")
    plt.tight_layout()
    plt.savefig(report_dir / "issue-distribution.png", dpi=150)
    plt.close()
    written.append("issue-distribution.png")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=issue_resolution, x="issue_type", y="resolution_time_minutes", color="#4e79a7")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Avg Resolution Time (minutes)")
    plt.xlabel("Issue Type")
    plt.title("Resolution Times by Issue Type")
    plt.tight_layout()
    plt.savefig(report_dir / "resolution-times.png", dpi=150)
    plt.close()
    written.append("resolution-times.png")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=customer_curve, x="ticket_id", y="satisfaction_rating", marker="o", color="#59a14f")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Satisfaction Rating")
    plt.xlabel("Ticket")
    plt.title("Customer Satisfaction by Ticket")
    plt.tight_layout()
    plt.savefig(report_dir / "customer-satisfaction.png", dpi=150)
    plt.close()
    written.append("customer-satisfaction.png")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=agent_df, x="agent_id", y="avg_resolution", color="#f28e2b")
    plt.ylabel("Avg Resolution Time (minutes)")
    plt.xlabel("Agent")
    plt.title("Agent Performance: Resolution")
    plt.tight_layout()
    plt.savefig(report_dir / "agent-performance-resolution.png", dpi=150)
    plt.close()
    written.append("agent-performance-resolution.png")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=agent_df, x="agent_id", y="avg_satisfaction", color="#76b7b2")
    plt.ylabel("Avg Satisfaction")
    plt.xlabel("Agent")
    plt.title("Agent Performance: Satisfaction")
    plt.tight_layout()
    plt.savefig(report_dir / "agent-performance-satisfaction.png", dpi=150)
    plt.close()
    written.append("agent-performance-satisfaction.png")

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.bar(agent_df["agent_id"], agent_df["avg_resolution"], color="#4e79a7", alpha=0.75)
    ax1.set_ylabel("Avg Resolution Time (minutes)", color="#4e79a7")
    ax1.set_xlabel("Agent")

    ax2 = ax1.twinx()
    ax2.plot(agent_df["agent_id"], agent_df["avg_satisfaction"], color="#e15759", marker="o", linewidth=2)
    ax2.set_ylabel("Avg Satisfaction", color="#e15759")

    plt.title("Agent Performance")
    fig.tight_layout()
    plt.savefig(report_dir / "agent-performance.png", dpi=150)
    plt.close(fig)
    written.append("agent-performance.png")

    return written
