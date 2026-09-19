"""Online segmented SAGE-MAS evolution.

Public entrypoints intentionally re-export the pipeline-wired implementation in
``online_evolution_legacy``. That loop calls ``SageEvolutionPipeline.run`` /
``repropose_organization`` for distill + org proposal.
"""

from sage_mas.online_evolution_legacy import (
    OnlineAlfWorldEvolution,
    OnlineEvolutionConfig,
    evaluation_trial_record,
    infer_dataset_split,
    segment_gamefiles,
    summarize_adaptation_history,
    summarize_dispatch,
    summarize_trials,
)
from sage_mas.online_webshop_evolution import OnlineWebShopEvolution

__all__ = [
    "OnlineAlfWorldEvolution",
    "OnlineWebShopEvolution",
    "OnlineEvolutionConfig",
    "evaluation_trial_record",
    "infer_dataset_split",
    "segment_gamefiles",
    "summarize_adaptation_history",
    "summarize_dispatch",
    "summarize_trials",
]
