#!/usr/bin/env python3
"""Generate a deterministic ONNX model for native terminal-pass selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, checker, helper


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def build_model() -> onnx.ModelProto:
    node = helper.make_node("Add", ["x", "y"], ["z"], name="probe_add")
    graph = helper.make_graph(
        [node],
        "native_terminal_probe",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4]),
        ],
        [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4])],
    )
    model = helper.make_model(
        graph,
        producer_name="ace-phantom-terminal-probe",
        opset_imports=[helper.make_operatorsetid("", 17)],
    )
    checker.check_model(model)
    return model


def main() -> int:
    arguments = parse_arguments()
    if arguments.output.suffix != ".onnx":
        raise SystemExit("output must use the .onnx suffix")
    model = build_model()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(model.SerializeToString(deterministic=True))
    print(f"generated {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
