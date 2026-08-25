"""Utilities to synthesize raw Verilog designs with Yosys into bit-level logic."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


def _normalize_path(path: str | Path) -> str:
    """Convert a filesystem path to a form safe for Yosys command-line arguments."""
    return str(Path(path)).replace("\\", "/").replace("\"", "\\\"")


def detect_top_module(verilog_path: str | Path) -> str:
    """Locate the module name in a Verilog file, preferring the file stem when present."""
    path = Path(verilog_path)
    if not path.exists():
        raise FileNotFoundError(f"Verilog design not found: {path}")

    text = path.read_text(encoding="utf-8")
    module_names = re.findall(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)\s*\(", text, flags=re.MULTILINE)
    if not module_names:
        raise ValueError(f"No module declarations found in {path}.")

    file_stem = path.stem
    if file_stem in module_names:
        return file_stem
    return module_names[0]


def build_yosys_script(
    input_path: str | Path,
    top_module: str,
    output_dir: str | Path,
    synth_path: str | Path,
    json_path: str | Path,
) -> str:
    """Construct the generic synthesis script executed by Yosys."""
    commands = [
        f"read_verilog -sv {_normalize_path(input_path)}",
        f"hierarchy -check -top {top_module}",
        "proc",
        "opt_expr",
        "wreduce",
        "alumacc",
        "share",
        "opt",
        "fsm",
        "memory -nomap",
        "techmap",
        "abc -g AND,OR,XOR,MUX",
        f"write_verilog -noattr {_normalize_path(synth_path)}",
        f"write_json {_normalize_path(json_path)}",
    ]
    return "\n".join(commands) + "\n"


def run_yosys_synthesis(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    top_module: str | None = None,
    yosys_bin: str = "yosys",
) -> dict[str, str | Path]:
    """Run Yosys synthesis for a single Verilog design file.

    Returns a dict with the synthetic output paths and resolved top module name.
    """
    source_path = Path(input_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Input design file does not exist: {source_path}")

    resolved_top = top_module or detect_top_module(source_path)
    output_path = Path(output_dir).expanduser().resolve() if output_dir is not None else source_path.parent / "yosys_synth"
    output_path.mkdir(parents=True, exist_ok=True)

    synth_verilog = output_path / f"{source_path.stem}.synth.v"
    json_ast = output_path / f"{source_path.stem}.json"

    yosys_executable = shutil.which(yosys_bin)
    if yosys_executable is None:
        raise FileNotFoundError(
            f"Yosys executable '{yosys_bin}' not found in PATH. Install Yosys or provide --yosys-bin."
        )

    script = build_yosys_script(
        input_path=source_path,
        top_module=resolved_top,
        output_dir=output_path,
        synth_path=synth_verilog,
        json_path=json_ast,
    )

    completed = subprocess.run(
        [yosys_executable, "-q", "-p", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "Unknown Yosys error"
        raise RuntimeError(f"Yosys synthesis failed for {source_path}: {stderr}")

    return {
        "top_module": resolved_top,
        "input_verilog": source_path,
        "output_dir": output_path,
        "synth_verilog": synth_verilog,
        "json_ast": json_ast,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthesize a Verilog design with Yosys into bit-level primitives.")
    parser.add_argument("input_file", type=Path, help="Path to the input Verilog source file.")
    parser.add_argument("--top", dest="top_module", default=None, help="Explicit top-module name. Auto-detected if omitted.")
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="Directory for generated synthesis artifacts.")
    parser.add_argument("--yosys-bin", default="yosys", help="Yosys executable path or command name.")
    args = parser.parse_args(argv)

    try:
        result = run_yosys_synthesis(
            input_path=args.input_file,
            output_dir=args.output_dir,
            top_module=args.top_module,
            yosys_bin=args.yosys_bin,
        )
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"ERROR: {exc}", file=None)
        return 1

    print(f"Top module: {result['top_module']}")
    print(f"Bit-level Verilog: {result['synth_verilog']}")
    print(f"Yosys JSON: {result['json_ast']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
