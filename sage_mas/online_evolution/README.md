# Online Evolution Module

**Runtime entrypoint:** package `__init__` re-exports
`online_evolution_legacy.OnlineAlfWorldEvolution`, which wraps
`SageEvolutionPipeline` for distill + org proposal each segment.

## Structure

```
online_evolution/
├── __init__.py              # Public exports → legacy (pipeline-wired)
└── README.md
```

An earlier incomplete modular split (coordinator/config/state_manager/probes/…)
was removed; `online_evolution_legacy.py` is the single live implementation of
the online loop. Any future refactor should re-split from that file against the
public API below.

## Usage

```python
from sage_mas.online_evolution import OnlineAlfWorldEvolution, OnlineEvolutionConfig

evolution = OnlineAlfWorldEvolution(
    llm_config=llm_config,
    sage_config=sage_config,
    output_root="logs/sage_mas/online",
)

result = evolution.run()
```
