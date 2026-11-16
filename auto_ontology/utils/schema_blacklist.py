# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-level exclusion list shared by the data and semantic layers.

A warehouse whose catalogs include system or debug schemas (e.g. GaussDWS
``gs_wlm_*`` / ``gs_blocklist_*`` / ``gs_dfg_*`` bookkeeping tables) must not
burn embedding/LLM budget — or fill either vector store with noise — over them.

The list is a comma-separated env value, ``AUTO_ONTOLOGY_SCHEMA_BLACKLIST``, of
exact ``catalog_schema.name`` matches. It is read once at import; change it by
setting the env var and restarting the service. Empty means no filtering, which
keeps the behaviour identical to before the list existed.
"""

from __future__ import annotations

import os

# Comma-separated, case-sensitive exact matches against the catalog schema name.
_BLACKLIST: frozenset[str] = frozenset(
    s.strip()
    for s in os.environ.get("AUTO_ONTOLOGY_SCHEMA_BLACKLIST", "").split(",")
    if s.strip()
)


def is_blacklisted_schema(schema_name: str | None) -> bool:
    """Whether *schema_name* is on the shared schema blacklist."""
    return bool(schema_name) and schema_name in _BLACKLIST


def blacklist() -> frozenset[str]:
    """The active blacklist (for logging)."""
    return _BLACKLIST


def blacklist_len() -> int:
    """Number of schemas on the active blacklist."""
    return len(_BLACKLIST)
