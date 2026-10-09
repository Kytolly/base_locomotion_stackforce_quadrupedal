"""Compatibility exports for the split curriculum package.

New task configurations must import from ``mdp.curricula`` so each sampling
axis remains independently auditable. This module stays for older scripts.
"""

from .curricula.terrain import curriculum_decisions, terrain_levels_by_episode_performance

__all__ = ["curriculum_decisions", "terrain_levels_by_episode_performance"]
