"""Assemble the selected-model V4 dataset with strict coverage checks."""

from __future__ import annotations

from typing import Sequence

from scripts.assemble_attribution_v3 import main as _assemble_main


def main(argv: Sequence[str] | None = None) -> int:
    """Use the generic strict assembler through a V4-named entry point."""

    return _assemble_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

