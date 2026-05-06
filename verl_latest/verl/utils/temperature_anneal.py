"""Cold-start temperature anneal for re_search rollouts.

Why
---
For base-model GRPO cold-start, the policy starts in a structural dead zone
(no ``<think>`` / ``<tool_call>`` tokens). Higher rollout sampling temperature
widens the support and lets the cascaded format rewards in
``r1_searcher_format`` actually fire on a few rollouts, providing a non-zero
gradient. Once structure is acquired, the temperature should decay back to the
production-time value to reduce reward variance.

What this anneals
-----------------
Only the **sampling temperature** at the agent-loop level
(``sampling_params["temperature"]`` in ``agent_loop.generate_sequences``).

Known bias (read this)
----------------------
The actor's PPO/GRPO log-prob recomputation in ``fsdp_workers.compute_log_prob``
hard-codes ``meta_info["temperature"] = self.config.rollout.temperature`` (the
init-time value). Annealing only the rollout sampling temperature therefore
introduces an importance-ratio bias: rollout draws come from
``π_old(.|s; T_t)`` but ratios are computed against ``π_θ(.|s; T_0)``.

In practice this is a small monotonic distortion if (a) the anneal range is
modest (e.g. ``T_start / T_end <= 1.5``), (b) it is monotonic and slow, and
(c) you keep ``actor.use_kl_loss=True`` to penalise drift. For wider ranges
mirror ``rollout.temperature`` into the worker config too (out of scope here).

Configuration (env vars, all optional)
--------------------------------------
``ROLLOUT_TEMPERATURE_ANNEAL``                 ``true``/``false`` (default ``false``)
``ROLLOUT_TEMPERATURE_START``                  start temperature (default = base)
``ROLLOUT_TEMPERATURE_END``                    end temperature   (default = base)
``ROLLOUT_TEMPERATURE_ANNEAL_STEPS``           # of training steps to interpolate over
``ROLLOUT_TEMPERATURE_ANNEAL_WARMUP_STEPS``    hold at ``START`` for this many steps first
``ROLLOUT_TEMPERATURE_ANNEAL_SCHEDULE``        ``linear`` | ``cosine`` (default ``linear``)
"""

from __future__ import annotations

import math
import os

_TRUE = {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def is_anneal_enabled() -> bool:
    return os.environ.get("ROLLOUT_TEMPERATURE_ANNEAL", "false").strip().lower() in _TRUE


def annealed_temperature(global_steps: int, base_temperature: float) -> float:
    """Return the rollout sampling temperature for ``global_steps``.

    No-op (returns ``base_temperature``) when:
      * ``ROLLOUT_TEMPERATURE_ANNEAL`` is not truthy, or
      * ``global_steps < 0`` (validation / unknown step).
    """
    if not is_anneal_enabled():
        return base_temperature
    if global_steps is None or global_steps < 0:
        return base_temperature

    start = _env_float("ROLLOUT_TEMPERATURE_START", base_temperature)
    end = _env_float("ROLLOUT_TEMPERATURE_END", base_temperature)
    total = max(_env_int("ROLLOUT_TEMPERATURE_ANNEAL_STEPS", 100), 1)
    warmup = max(_env_int("ROLLOUT_TEMPERATURE_ANNEAL_WARMUP_STEPS", 0), 0)
    schedule = os.environ.get("ROLLOUT_TEMPERATURE_ANNEAL_SCHEDULE", "linear").strip().lower()

    if global_steps < warmup:
        return max(start, 1e-6)

    progress = (global_steps - warmup) / float(total)
    progress = min(max(progress, 0.0), 1.0)

    if schedule == "cosine":
        progress = 0.5 * (1.0 - math.cos(math.pi * progress))

    value = start + (end - start) * progress
    return max(value, 1e-6)
