"""Utilities to synthesize raw Verilog designs with Yosys into bit-level logic."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


def _normalize_path(path: str | Path) -> str:
    """Convert a filesystem path to a form safe for Yosys command-line arguments."""
    return str(Path(path)).replace("\\", "/").replace('"', '\\"')


def detect_top_module(verilog_path: str | Path) -> str:
    """Locate the module name in a Verilog file, preferring the file stem when present."""
    path = Path(verilog_path)
    if not path.exists():
        raise FileNotFoundError(f"Verilog design not found: {path}")

    text = path.read_text(encoding="utf-8")
    module_names = re.findall(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)\s*[(;#]", text, flags=re.MULTILINE)
    if not module_names:
        raise ValueError(f"No module declarations found in {path}.")

    file_stem = path.stem
    if file_stem in module_names:
        return file_stem
    return module_names[0]


def find_yosys_binary(yosys_bin: str = "yosys") -> Path:
    """Discover the Yosys executable across PATH, OSS_CAD_SUITE, and known default locations."""
    # 1. Direct path check
    direct_candidate = Path(yosys_bin).expanduser().resolve()
    if direct_candidate.is_file():
        return direct_candidate

    # 2. Check system PATH
    found = shutil.which(yosys_bin)
    if found:
        return Path(found).resolve()

    # 3. Check OSS_CAD_SUITE environment variable
    oss_root = os.environ.get("OSS_CAD_SUITE")
    if oss_root:
        bin_dir = Path(oss_root) / "bin"
        exe_name = "yosys.exe" if os.name == "nt" else "yosys"
        cand = bin_dir / exe_name
        if cand.is_file():
            return cand.resolve()

    # 4. Known default locations on Windows
    known_paths = [
        Path(r"D:\oss-cad-suite-windows-x64-20260824\oss-cad-suite-windows-x64-20260824\oss-cad-suite\bin\yosys.exe"),
        Path(r"C:\oss-cad-suite\bin\yosys.exe"),
        Path(r"D:\oss-cad-suite\bin\yosys.exe"),
    ]
    for kp in known_paths:
        if kp.is_file():
            return kp.resolve()

    raise FileNotFoundError(
        f"Yosys executable '{yosys_bin}' not found in PATH or known locations. "
        "Install Yosys or set OSS_CAD_SUITE environment variable."
    )


def prepare_yosys_env(yosys_executable: Path) -> dict[str, str]:
    """Prepare process environment so dependent DLLs and helper binaries (abc, python) are reachable."""
    env = dict(os.environ)
    bin_dir = yosys_executable.parent
    root_dir = bin_dir.parent
    lib_dir = root_dir / "lib"

    path_additions = [str(bin_dir)]
    if lib_dir.is_dir():
        path_additions.append(str(lib_dir))

    current_path = env.get("PATH", "")
    prefix = ";".join(path_additions) if os.name == "nt" else ":".join(path_additions)
    delimiter = ";" if os.name == "nt" else ":"
    env["PATH"] = f"{prefix}{delimiter}{current_path}"
    return env


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
        "clean",
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

    yosys_executable = find_yosys_binary(yosys_bin)
    env = prepare_yosys_env(yosys_executable)

    script = build_yosys_script(
        input_path=source_path,
        top_module=resolved_top,
        output_dir=output_path,
        synth_path=synth_verilog,
        json_path=json_ast,
    )

    completed = subprocess.run(
        [str(yosys_executable), "-q", "-p", script],
        capture_output=True,
        text=True,
        check=False,
        env=env,
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
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Top module: {result['top_module']}")
    print(f"Bit-level Verilog: {result['synth_verilog']}")
    print(f"Yosys JSON: {result['json_ast']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
