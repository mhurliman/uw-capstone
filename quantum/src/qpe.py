from absl import (app, flags)
import cirq
import math
import matplotlib.pyplot as plt
import numpy as np
import qft

FLAGS = flags.FLAGS
flags.DEFINE_integer('iterations', 1, 'The number of experiment iterations to garner a meaningful sample of the distribution.')
flags.DEFINE_integer('precision', 10, 'Bit precision of the phase estimation.')

def run_quantum(iterations, precision):

    # Set up unitary operation and eigenstate
    eigenstate_bits = cirq.LineQubit(precision + 1)
    unitary_gate = cirq.Rz(rads=math.pi / 8)

    # Set up output phase bits
    phase_bits = cirq.LineQubit.range(precision)

    # Set initial states
    circuit = cirq.Circuit()
    circuit.append([cirq.H(x) for x in phase_bits])
    circuit.append(cirq.X(eigenstate_bits))

    # Unitary iteration
    for i in range(precision):
        circuit.append(unitary_gate(eigenstate_bits).controlled_by(phase_bits[i])**(2**i))

    # Convert phase to qubit state
    circuit.append(qft.iQFT(phase_bits), strategy=cirq.InsertStrategy.NEW)

    #simulator = cirq.Simulator()
    #result = simulator.run(circuit, repetitions=iterations)

    wf = cirq.final_state_vector(circuit)
    mix = cirq.partial_trace_of_state_vector_as_mixture(wf, keep_indices=range(precision))

    for p, c in mix:
        print(f'{p**2:%} {cirq.dirac_notation(c)}')

    # Serialize the network structure to file
    with open('circuits\circuit_qpe.txt', 'wt', encoding='utf-8') as f:
        f.write(str(circuit))


def main(argv):
    run_quantum(FLAGS.iterations, FLAGS.precision)

if __name__ == '__main__':
    app.run(main)
