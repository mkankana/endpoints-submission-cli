# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Benchmarks the reference implementation collects in Offline only (§3.2, §5.7).

§3.2 makes the reference implementation the authority for per-model specifications,
and its required-scenarios table lists some benchmarks as Offline-only — upstream
``speech2text`` is ``["Offline"]``. Such a benchmark has no fixed-concurrency Pareto
frontier to collect, so the rules that describe one do not apply to it: §5.3's seven
points, §5.4's Ultra Low Concurrency coverage and §§3–6's three concurrency regions
all presuppose a curve, and §5.5's boundary algorithm needs a ``C_max`` above 32 that
an Offline-only frontier has no points to place.

The benchmark decides, never the submission. A submitter cannot declare their way out
of the Pareto requirement by shipping one ``max_throughput`` point and calling the
frontier Offline-only: the lookup key is the model name, and ``model-name-valid``
already requires that name to be one of §3.2's canonical spellings.

Like ``checker._ALLOWED_MODEL_NAMES``, this is round-scoped reference data. §3.2
publishes the model set "at least 6 weeks before the submission round opens", so a new
Offline-only benchmark should ideally arrive as data rather than as a code change —
``data/seed_sets.yaml`` is the pattern to follow when that is worth doing.
"""

from __future__ import annotations

__all__ = ["OFFLINE_ONLY_MODELS", "is_offline_only"]

#: Canonical (§8.1 directory) names of the Offline-only benchmark models.
#:
#: ``whisper-large-v3`` is upstream ``speech2text``, whose required-scenarios entry is
#: ``["Offline"]``. It is also non-streaming — it serves on
#: ``/v1/audio/transcriptions``, which has no SSE mode — so it reports neither TTFT nor
#: TPOT. §5.7.1's dedicated-Offline exemption from the latency metrics is therefore the
#: only reading under which any of its points validate, which is a second, independent
#: reason its points cannot be ordinary fixed-concurrency ones: §5.7.2 is explicit that
#: the exemption "appl[ies] to a dedicated Offline run, not to an elected point".
#:
#: Matched exactly rather than by keyword fragment (as ``accuracy_targets`` does, to
#: cover fine-tunes): waiving the Pareto requirement is the permissive branch, so a
#: near-miss name must fall through to the strict one.
OFFLINE_ONLY_MODELS = frozenset({"whisper-large-v3"})


def is_offline_only(model: str) -> bool:
    """Whether *model* is a benchmark collected in Offline only.

    Args:
        model: Model directory name (§8.1), e.g. ``"whisper-large-v3"``.
    """
    return model in OFFLINE_ONLY_MODELS
