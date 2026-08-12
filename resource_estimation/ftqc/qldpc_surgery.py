# Copyright 2026 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import collections
import typing

import cirq
import numpy as np
from qldpc import codes
from qldpc.objects import Pauli

import resource_estimation.ftqc.lattice_surgery_primitives as lsp
from resource_estimation.ftqc.stim_functions import count_stim_resources


class LogicalPPMResourceCost(typing.TypedDict):
    """Physical resources extracted from a qLDPC joint-PPM circuit."""

    gate_cost: collections.Counter[cirq.Gate]  # Total operations, grouped by gate type.
    moment_cost: collections.Counter[cirq.Gate]  # Parallel circuit layers by gate type.
    num_physical_qubits: int  # Peak footprint of the generated Stim circuit.


def _logical_operator(
    code: codes.CSSCode,
    pauli: Pauli,
    logical_index: int,
) -> np.ndarray:
    """Return the physical support of one logical Pauli representative.

    qLDPC returns one binary support vector for each of the ``k`` logical qubits
    encoded by ``code``.  Entry ``j`` of a support vector is nonzero when the
    selected logical operator acts on physical data qubit ``j``.

    Args:
        code: CSS code containing the requested logical qubit.
        pauli: Logical Pauli type to retrieve, currently ``Pauli.X`` or
            ``Pauli.Z``.
        logical_index: Zero-based index of the logical qubit within the code.

    Returns:
        A length-``n`` binary vector supporting the selected logical operator.

    Raises:
        TypeError: If ``logical_index`` is not an integer.
        ValueError: If ``logical_index`` is negative or outside the code's
            logical-qubit range.
    """
    # bool is an int subclass in Python, but True/False are not meaningful
    # logical-qubit indices for this interface.
    if not isinstance(logical_index, int) or isinstance(logical_index, bool):
        raise TypeError("logical indices must be integers")
    # Reject negative values instead of allowing NumPy's negative indexing.
    if logical_index < 0:
        raise ValueError("logical indices must be nonnegative")

    # get_logical_ops returns one physical-support vector per encoded logical
    # qubit for the requested Pauli type.
    logical_operators = code.get_logical_ops(pauli)
    if logical_index >= len(logical_operators):
        raise ValueError(
            f"logical index {logical_index} is out of range for a code with "
            f"{len(logical_operators)} logical qubits",
        )
    return logical_operators[logical_index]


def logical_ppm_resource_cost(
    operation: cirq.Operation,
    left_code: codes.CSSCode,
    right_code: codes.CSSCode,
    *,
    rounds: int,
    left_logical_index: int = 0,
    right_logical_index: int = 0,
) -> LogicalPPMResourceCost:
    """Build and count a qLDPC surgery circuit for a logical joint measurement.

    The two Cirq qubits identify the logical operands, while ``left_code`` and
    ``right_code`` provide their physical code definitions.  The corresponding
    logical indices select which encoded mode to measure when either code has
    ``k > 1``.

    Args:
        operation: A two-qubit operation created from :class:`LogicalPPM`.
            Its gate stores whether this is an ``XX`` or ``ZZ`` measurement.
        left_code: qLDPC CSS code for the operation's first logical operand.
        right_code: qLDPC CSS code for the operation's second logical operand.
        rounds: Positive number of syndrome-extraction rounds in the generated
            joint-measurement circuit.
        left_logical_index: Logical-qubit index within ``left_code``.
        right_logical_index: Logical-qubit index within ``right_code``.

    Returns:
        Serial gate counts, parallel moment counts, and the physical-qubit
        footprint of the generated Stim circuit.

    Raises:
        TypeError: If the operation, codes, rounds, or logical indices have the
            wrong type.
        ValueError: If rounds or a logical index is outside its valid range.
        ImportError: If the installed qLDPC version lacks the surgery
            resource-circuit API.

    This cost excludes architecture-specific movement, post-detachment syndrome
    extraction, and measurement-conditioned Pauli-frame updates.
    """
    if not isinstance(operation.gate, lsp.LogicalPPM):
        raise TypeError("operation must use a LogicalPPM gate")
    if not isinstance(left_code, codes.CSSCode) or not isinstance(right_code, codes.CSSCode):
        raise TypeError("left_code and right_code must be qLDPC CSSCode objects")
    if not isinstance(rounds, int) or isinstance(rounds, bool):
        raise TypeError("rounds must be an integer")
    if rounds < 1:
        raise ValueError("rounds must be positive")

    # Import lazily so the rest of the estimator remains importable with a qLDPC
    # release that does not yet contain this surgery API.
    try:
        from qldpc.circuits.surgery import (
            build_bridge,
            build_gadget,
            build_joint_ppm_resource_circuit,
        )
    except ImportError as ex:  # pragma: no cover - requires an incompatible qLDPC installation
        raise ImportError(
            "LogicalPPM costing requires a qLDPC version with the surgery resource-circuit API.",
        ) from ex

    # Cirq's GateOperation retains the LogicalPPM object passed to .on(...), so
    # its stored string selects the corresponding qLDPC Pauli basis here.
    pauli = Pauli.X if operation.gate.pauli_product == "XX" else Pauli.Z

    # Select a physical representative of the requested logical Pauli in each
    # code, then turn each code/operator pair into a qLDPC surgery gadget.
    left_operator = _logical_operator(left_code, pauli, left_logical_index)
    right_operator = _logical_operator(right_code, pauli, right_logical_index)
    left_gadget = build_gadget(left_code, left_operator, basis=pauli)
    right_gadget = build_gadget(right_code, right_operator, basis=pauli)

    # The bridge joins the two gadgets.  qLDPC then emits the complete Stim
    # circuit for the requested number of syndrome rounds.
    bridge = build_bridge(left_gadget, right_gadget)
    resource = build_joint_ppm_resource_circuit(
        left_gadget,
        right_gadget,
        bridge,
        rounds=rounds,
    )
    # Translate the physical Stim circuit into the estimator's usual serial and
    # parallel Cirq-gate counters.
    resources = count_stim_resources(resource.circuit)
    return {
        "gate_cost": resources["serial"],
        "moment_cost": resources["parallel"],
        "num_physical_qubits": resource.circuit.num_qubits,
    }
