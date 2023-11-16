from absl import (app, flags)
import cirq
import math
import matplotlib.pyplot as plt
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_integer('iterations', 1, 'The number of experiment iterations to garner a meaningful sample of the distribution.')
flags.DEFINE_integer('data', 2, 'Data to encode into the qubit phase.')

def run_quantum(iterations, data):
    data = min(data, 3)

    def entangler_circuit(bits):
        yield cirq.H(bits[0])
        yield cirq.X(bits[1]).controlled_by(bits[0])

    bits = cirq.LineQubit.range(2)

    circuit = cirq.Circuit()
    circuit.append(entangler_circuit(bits))

    if (data & 0x1):
        circuit.append(cirq.X(bits[0]))
    if (data & 0x2):
        circuit.append(cirq.Z(bits[0]))

    circuit.append(cirq.X(bits[1]).controlled_by(bits[0]))
    circuit.append(cirq.H(bits[0]))
    #circuit.append(cirq.M(bits[0:2]))

    #simulator = cirq.Simulator()
    #result = simulator.run(circuit, repetitions=iterations)

    wf = cirq.final_state_vector(circuit)
    mix = cirq.partial_trace_of_state_vector_as_mixture(wf, keep_indices=[0, 1])

    for p, c in mix:
        print(f'{p**2:%} {cirq.dirac_notation(c)}')

        
    # Serialize the network structure to file
    with open('superdense.txt', 'wt', encoding='utf-8') as f:
        f.write(str(circuit))



def main(argv):
    run_quantum(FLAGS.iterations, FLAGS.data)

if __name__ == '__main__':
    """Entrypoint"""
    app.run(main)
