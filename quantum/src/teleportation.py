from absl import (app, flags)
import cirq
import math
import matplotlib.pyplot as plt
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_integer('iterations', 1, 'The number of experiment iterations to garner a meaningful sample of the distribution.')
flags.DEFINE_float('p', 0.5, 'Probability of |0> state of teleported quantum state.')

def run_quantum(iterations, p):
    
    theta = math.acos(math.sqrt(p))*2

    def entangler_circuit(bits):
        yield cirq.H(bits[0])
        yield cirq.X(bits[1]).controlled_by(bits[0])


    circuit = cirq.Circuit()
    
    bits = cirq.LineQubit.range(3)
    teleport_bit = bits[0]
    shared_bits = bits[1:3]

    circuit.append(entangler_circuit(shared_bits))
    circuit.append(cirq.Ry(rads=theta)(teleport_bit))

    circuit.append(cirq.X(shared_bits[0]).controlled_by(teleport_bit))
    circuit.append(cirq.H(teleport_bit))

    #circuit.append(cirq.M(bits[0:2]))

    #simulator = cirq.Simulator()
    #result = simulator.run(circuit, repetitions=iterations)

    wf = cirq.final_state_vector(circuit)
    mix = cirq.partial_trace_of_state_vector_as_mixture(wf, keep_indices=[0, 1])

    for p, c in mix:
        print(f'{p**2:%} {cirq.dirac_notation(c)}')

        
    # Serialize the network structure to file
    with open('teleport.txt', 'wt', encoding='utf-8') as f:
        f.write(str(circuit))



def main(argv):
    run_quantum(FLAGS.iterations, FLAGS.p)

if __name__ == '__main__':
    """Entrypoint"""
    app.run(main)
