"""Re-export mission JSON loader (implementation in ``cli.mission_loader``)."""

from src.research.langchain_agent.cli.mission_loader import load_mission_from_file

__all__ = ["load_mission_from_file"]
