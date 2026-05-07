from pathlib import Path

import pandas as pd
try:
    from crewai_tools import BaseTool
except ImportError:
    from crewai.tools import BaseTool


class TicketStatsTool(BaseTool):
    """
    CrewAI tool that reads the support tickets CSV and returns a plain-text
    statistical summary for use by the ticket_triage_agent during analysis.

    The tool requires no arguments — it resolves the data file path automatically
    relative to the package root.
    """
    name: str = "ticket_stats_tool"
    description: str = "Summarize support ticket CSV metrics for triage and planning."

    def _run(self) -> str:
        """
        Execute the tool: load the CSV and compute ticket statistics.

        No parameters are required. The CSV path is resolved automatically as:
        <package_root>/data/support_tickets_data.csv

        Returns
        -------
        str
            A plain-text summary with the following metrics:
            - Total number of tickets
            - Average resolution time in minutes
            - Average customer satisfaction rating (out of 5)
            - Ticket count broken down by priority level
            - Ticket count broken down by issue type

            Returns an error string if the CSV file is not found.
        """
        project_root = Path(__file__).resolve().parents[3]
        csv_path = project_root / "data" / "support_tickets_data.csv"
        if not csv_path.exists():
            return f"Data file not found: {csv_path}"

        df = pd.read_csv(csv_path)
        total = len(df)
        avg_resolution = round(float(df["resolution_time_minutes"].mean()), 2)
        avg_satisfaction = round(float(df["satisfaction_rating"].mean()), 2)
        by_priority = (
            df["priority"].value_counts(dropna=False).to_dict()
        )
        by_issue = df["issue_type"].value_counts(dropna=False).to_dict()

        return (
            f"Total tickets: {total}\n"
            f"Average resolution time: {avg_resolution} minutes\n"
            f"Average satisfaction: {avg_satisfaction} / 5\n"
            f"Tickets by priority: {by_priority}\n"
            f"Tickets by issue: {by_issue}"
        )
