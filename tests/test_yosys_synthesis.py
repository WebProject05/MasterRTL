from pathlib import Path

from src.data_prep.yosys_synthesis import build_yosys_script, detect_top_module


def test_detect_top_module_prefers_filename_match(tmp_path: Path) -> None:
    design = tmp_path / "adder.v"
    design.write_text(
        """
module helper(input_a, input_b, sum);
    input input_a;
    input input_b;
    output sum;
    assign sum = input_a ^ input_b;
endmodule

module adder(input_a, input_b, sum);
    input input_a;
    input input_b;
    output sum;
    assign sum = input_a + input_b;
endmodule
""".strip(),
        encoding="utf-8",
    )

    assert detect_top_module(design) == "adder"


def test_build_yosys_script_contains_required_passes() -> None:
    script = build_yosys_script(
        input_path="/tmp/design.v",
        top_module="top",
        output_dir="/tmp/out",
        synth_path="/tmp/out/design.synth.v",
        json_path="/tmp/out/design.json",
    )

    required_steps = [
        "read_verilog -sv /tmp/design.v",
        "hierarchy -check -top top",
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
        "write_verilog -noattr /tmp/out/design.synth.v",
        "write_json /tmp/out/design.json",
    ]

    for step in required_steps:
        assert step in script
