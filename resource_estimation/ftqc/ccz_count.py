from collections import Counter
from typing import Literal
import cirq

def count_ccz(arch: res.architecture.Architecture, reps: int, depth: int) -> dict[Literal['op_time', 'moment_cost', 'gate_cost'], Counter[cirq.Gate, int]]:
    # Folowing Fig 5 in http://arxiv.org/abs/1812.01238
    # We include a Logical Reset layer
    # A Hadamard moment
    # Next is a layer of 10 Logical CNOTs
    # Next is a layer of T gates
    #     One way to handle is to do physical T and grow
    #     Another way to handle is to perform a layer of CNOTs on previously distilled qubits, possibly with corrections
    #     Another way is to perform a layer of CNOTs on previously cultivated qubits, possibly with corrections (But this is a difficult problem so I'll approach it later)
    # Next is a layer of Logical Hadamard gates
    # Finally a layer of logical Measurement gates is performed
    # Based on the results of the measurement, the process is repeated or deemed complete
    gate_to_cost = {
        cirq.H: '_h_cost',
        cirq.CNOT: '_cnot_cost'
    }
    logical_moment_cost = ccz_moments(reps=reps, depth=depth)
    logical_gate_cost = ccz_gates(reps=reps, depth=depth)
    physical_moment_cost = Counter()
    for cirq_gate, num_occurrences in logical_moment_cost.items():
        if has_cost(cirq_gate):
            physical_moment_cost += arch.op_cost(cirq_gate)
        else:
            physical_moment_cost += do_something_else(cirq_gate)
    for cirq_gate, num_occurrences in logical_gate_cost.items():
        if has_cost(cirq_gate):
            physical_gate_cost += arch.op_cost(cirq_gate)
        else:
            physical_gate_cost += do_something_else(cirq_gate)
    op_time = arch.total_time(physical_moment_cost)
    return {'op_time': op_time, 'moment_cost': physical_moment_cost, 'gate_cost': physical_gate_cost}

def ccz_moments(reps: int, depth: int):
    if depth == 0:
        cost = Counter({
            cirq.ResetChannel: 1 * reps,
            cirq.H: 2 * reps,
            cirq.CNOT: 10 * reps,
            cirq.T: 1 * reps,
            cirq.MeasurmentGate: 1 * reps
        })
        return cost
    cost = Counter({
        cirq.CNOT: 1 * reps,
        cirq.MeasurementGate: 1 * reps,
        cirq.S: 1 * reps,  # Conditional on measurement outcome 
    })
    cost += ccz_moments(reps*reps, depth-1)
    return cost

def ccz_gates(reps: int, depth: int):
    if depth == 0:
        cost = Counter({
            cirq.ResetChannel: 15 * reps,
            cirq.H: 16 * reps,
            cirq.CNOT: 23 * reps,
            cirq.T: 8 * reps,
            cirq.MeasurementGate: 12 * reps,
        })
        return cost
    else:
        cost = Counter({
            cirq.ResetChannel: 15 * reps,
            cirq.H: 16 * reps,
            cirq.CNOT: 23 * reps,
            cirq.MeasurementGate: 12 * reps,
        })
        cost += ccz_gates(reps*reps, depth-1)
        
