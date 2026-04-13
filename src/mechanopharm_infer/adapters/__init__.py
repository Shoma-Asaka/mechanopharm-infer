"""Thin dataset adapters for literature-derived mechanopharmacology tables.

These adapters intentionally do only three things:
1. absorb small column-name differences,
2. coerce mechanics labels to numeric values when needed,
3. attach dataset/system/assay metadata before routing into the common
   schema + preprocessing pipeline.
"""

from .jem_tnbc import (
    DEFAULT_JEM_MECHANICS_MAP,
    DEFAULT_JEM_METADATA,
    prepare_jem_endpoint,
    prepare_jem_timecourse,
)
from .kalli_pdac import DEFAULT_KALLI_METADATA, prepare_kalli_endpoint, prepare_kalli_timecourse
from .novak_ovarian import DEFAULT_NOVAK_METADATA, prepare_novak_endpoint, prepare_novak_timecourse

__all__ = [
    "DEFAULT_JEM_MECHANICS_MAP",
    "DEFAULT_JEM_METADATA",
    "DEFAULT_NOVAK_METADATA",
    "DEFAULT_KALLI_METADATA",
    "prepare_jem_endpoint",
    "prepare_jem_timecourse",
    "prepare_novak_endpoint",
    "prepare_novak_timecourse",
    "prepare_kalli_endpoint",
    "prepare_kalli_timecourse",
]
