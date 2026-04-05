# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Live W&B Weave logging for rollout JSONL rows (same payload as each JSON line).

Enable with ``trainer.weave_rollout_live=true`` or ``WEAVE_ROLLOUT_LIVE=1``.
Requires ``wandb`` in ``trainer.logger`` and ``pip install weave``.

Each row is logged as a Weave call ``verl_rollout_jsonl_row`` with inputs equal to the
JSON object written to disk, and associated with the active ``wandb.run`` via
``set_wandb_run_context`` (see W&B Weave docs).
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_client: Any = None
_init_attempted: bool = False

WEAVE_OP_NAME = "verl_rollout_jsonl_row"


def is_enabled(config: dict | None) -> bool:
    if os.environ.get("WEAVE_ROLLOUT_LIVE", "").lower() in ("1", "true", "yes"):
        return True
    if config and config.get("trainer", {}).get("weave_rollout_live"):
        return True
    return False


def _sanitize(obj: Any) -> Any:
    """Make values safe for Weave / JSON-like payloads."""
    if obj is None:
        return None
    if isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(x) for x in obj]
    return str(obj)


def init_from_tracking_config(config: dict | None) -> None:
    """Call once after ``wandb.init`` (e.g. from ``Tracking.__init__``)."""
    global _client, _init_attempted
    _init_attempted = True
    if not is_enabled(config):
        return
    try:
        import wandb
        import weave
    except ImportError:
        logger.warning("weave_rollout_live is enabled but `weave` is not installed; pip install weave")
        return
    if wandb.run is None:
        logger.warning("weave_rollout_live: no wandb.run; skipping Weave init")
        return
    try:
        entity = os.environ.get("WANDB_ENTITY", "").strip()
        proj = (config or {}).get("trainer", {}).get("project_name", "verl_examples")
        path = f"{entity}/{proj}" if entity else proj
        _client = weave.init(path)
        _client.set_wandb_run_context(run_id=wandb.run.id, step=0)
    except Exception as e:
        logger.warning("Weave init failed (training continues): %s", e)
        _client = None


def log_jsonl_rows(global_step: int, rows: list[dict[str, Any]]) -> None:
    """Log each row as a Weave call; same dicts as each JSONL line."""
    global _client
    if not rows:
        return
    if _client is None:
        if not _init_attempted:
            logger.debug("weave_rollout: init not attempted; skipping")
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    try:
        _client.set_wandb_run_context(run_id=wandb.run.id, step=int(global_step))
        for row in rows:
            payload = _sanitize(row)
            call = _client.create_call(WEAVE_OP_NAME, inputs=payload, use_stack=False)
            _client.finish_call(call, output={"ok": True})
        _client.flush()
    except Exception as e:
        logger.warning("Weave rollout logging failed (training continues): %s", e)


def finish() -> None:
    """Flush Weave client on shutdown."""
    global _client
    if _client is None:
        return
    try:
        _client.finish(use_progress_bar=False)
    except Exception as e:
        logger.debug("Weave finish: %s", e)
    _client = None
