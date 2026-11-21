#!/usr/bin/env python3
"""Apply the frozen cooling/ownership engine to the angular Ax campaign.

Only the named Ax supervisor is added. The same script may run its explicitly
allowlisted ask/tell helpers; neural stages use the existing angular trainer.
No base guard file or training implementation is modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys

import thermal_guard_cool_v2 as base


AX_SUPERVISOR = 'run_ax_geometry_search.py'
AX_STAGE_SCRIPTS = frozenset({AX_SUPERVISOR, 'train_geometric_variants.py'})


class AxCoolingGuard(base.CoolingGuard):
    def __init__(self, *args, policy_metadata=None, **kwargs):
        metadata = dict(policy_metadata or {})
        metadata.update(ax_guard_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        ax_supervisor=AX_SUPERVISOR)
        super().__init__(*args, policy_metadata=metadata, **kwargs)


@contextmanager
def bound_ax_ownership():
    """Bind just the additional supervisor, restoring globals after library use."""
    old_supervisors = base.SUPERVISORS
    old_engine_supervisors = base.engine.SUPERVISORS
    old_trainers = base.engine.TRAINERS
    old_defaults = base.DEFAULT_STAGE_SCRIPTS
    old_guard = base.CoolingGuard
    try:
        base.SUPERVISORS = old_supervisors | {AX_SUPERVISOR}
        base.engine.SUPERVISORS = set(old_engine_supervisors) | {AX_SUPERVISOR}
        base.engine.TRAINERS = set(old_trainers)
        # Ask/tell is enabled explicitly by --allow-stage-script, while the
        # default neural stage stays within this one geometric architecture.
        base.DEFAULT_STAGE_SCRIPTS = frozenset({'train_geometric_variants.py'})
        base.CoolingGuard = AxCoolingGuard
        yield
    finally:
        base.SUPERVISORS = old_supervisors
        base.engine.SUPERVISORS = old_engine_supervisors
        base.engine.TRAINERS = old_trainers
        base.DEFAULT_STAGE_SCRIPTS = old_defaults
        base.CoolingGuard = old_guard


def validate_extra_stages(arguments):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--allow-stage-script', action='append', default=[])
    known, _ = parser.parse_known_args(arguments)
    if not set(known.allow_stage_script) <= AX_STAGE_SCRIPTS:
        raise ValueError('Ax cooling permits only the angular trainer and explicit Ax ask/tell helper')


def main():
    validate_extra_stages(sys.argv[1:])
    with bound_ax_ownership():
        return base.main()


if __name__ == '__main__':
    main()
