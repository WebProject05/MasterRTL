"""Batch utility for synthesizing a directory of Verilog files with Yosys."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Iterable, Sequence

from src.data_prep.yosys_synthesis import run_yosys_synthesis


logger = logging.getLogger(__name__)


def iter_verilog_files(input_dir: Path, recursive: bool = False) -> list[Path]:
    """Return a sorted list of Verilog source files under the given directory."""
    patterns = ("*.v", "*.sv")
    if recursive:
        matches = [path for pattern in patterns for path in input_dir.rglob(pattern)]
    else:
        matches = [path for pattern in patterns for path in input_dir.glob(pattern)]
    return sorted({path.resolve() for path in matches})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-synthesize Verilog files with Yosys and emit bit-level .synth.v "
            "plus JSON AST artifacts."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing Verilog inputs.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated synthesis artifacts. Defaults to <input_dir>/yosys_synth.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively for Verilog files.",
    )
    parser.add_argument(
        "--yosys-bin",
        default="yosys",
        help="Yosys executable to invoke (default: yosys).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    output_base = (args.output_dir or input_dir / "yosys_synth").expanduser().resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    files = iter_verilog_files(input_dir, recursive=args.recursive)
    if not files:
        print(f"No Verilog files found under {input_dir}.")
        return 1

    total_start = time.perf_counter()
    total = 0
    succeeded = 0
    failed = 0

    for design_path in files:
        total += 1
        started = time.perf_counter()
        rel_path = design_path.relative_to(input_dir)
        print(f"[{total}/{len(files)}] Processing {rel_path}...")

        try:
            result = run_yosys_synthesis(
                input_path=design_path,
                output_dir=output_base,
                yosys_bin=args.yosys_bin,
            )
            elapsed = time.perf_counter() - started
            print(
                f"  SUCCESS: top={result['top_module']} | "
                f"synth={result['synth_verilog'].name} | "
                f"json={result['json_ast'].name} | elapsed={elapsed:.2f}s"
            )
            succeeded += 1
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - started
            print(f"  ERROR: {design_path.name} | elapsed={elapsed:.2f}s | {exc}")
            logger.exception("Yosys synthesis failed for %s", design_path)
            failed += 1

    total_elapsed = time.perf_counter() - total_start
    print("\nBatch summary:")
    print(f"  scanned: {total}")
    print(f"  succeeded: {succeeded}")
    print(f"  failed: {failed}")
    print(f"  total runtime: {total_elapsed:.2f}s")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
