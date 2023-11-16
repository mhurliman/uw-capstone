from absl import (app, flags)
import cirq
import math
import matplotlib.pyplot as plt
import numpy as np


# Command line flags
FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'quantum', enum_values=['classical', 'quantum'], case_sensitive=False, help='Specifies whether to run a "classical" or "quantum" walk.')
flags.DEFINE_integer('iterations', 1000, 'The number of experiment iterations to garner a meaningful sample of the distribution.')
flags.DEFINE_integer('num_steps', 50, 'The number of steps to take within a single walk.')
flags.DEFINE_integer('start', 0, lower_bound=-1000, upper_bound=1000, help='The X-axis starting position of the walk.')
flags.DEFINE_float('p', 0.5, 'Probability of the coin flip to evaluate a walk to the right.')

# Helper routine to get the number of bits a number requires
def bit_count(x):
    return (x-1).bit_length()


def line_classical(iterations, num_steps=50, start=0, p=0.5):
    """Performs a classical walk on the line (1-D lattice)
    :param iterations: number of times to perform the walk
    :param num_steps: number of steps to take in the walk
    :param start: starting X position of the walk (the walk is unvariant under translation, so it's just a translation of the results)
    :param p: probability of a step in the +X direction
    """

    def walk(num_steps, start=0, p=0.5):
        """Performs the classical walk over a line"""
        position = 0

        # Walk according to a simple coin flip
        for _ in range(num_steps):
            step = np.random.choice(2, 1, p=[1-p, p])[0] # Coin flip
            step = step * 2 - 1                          # Simple remapping of 0,1 to -1,1
            position += step                             # Take the step

        return position

    # Initialize the result array to zeros
    results = np.zeros(num_steps * 2 + 1, np.int32)

    # Perform the walks
    for i in range(iterations):
        final = walk(num_steps, start, p)
        results[num_steps+final] += 1

    # Graph the results
    plt.bar(range(-num_steps + start, num_steps + start + 1), [n/iterations for n in results])
    plt.show()



def line_quantum(iterations, num_steps=30, start=0, p=0.5):
    """
    Performs a coined quantum walk on a line (1-D lattice)

    :param iterations: Number of times to perform the walk
    :param N: Number of steps to take in the walk
    :param start: Starting X position of the walk (the walk is unvariant under translation, so it's just a translation of the results)
    :param p: Probability of a step in the +X direction
    """

    # Given we walk N steps we need enough states to walk N steps in either direction from x=0 
    # -N to N -> 2N + 1 states -> log2(2N + 1) qubits
    num_qubits = bit_count(num_steps * 2 + 1)
    
    # Coin qubit is our last indexable bit
    COIN_BIT = num_qubits

    # Rotation about bloch X-axis by theta
    # Probability is dictated by relative angle to |0>
    theta = math.acos(math.sqrt(p))*2

    def init(bits):
        """Initializes the position and coin qubits to starting state"""

        # Start at state which is halfway between our bit range (0b100...0) (otherwise we'll over/underflow)
        yield cirq.X(cirq.LineQubit(0)) 

        # Begin coin in superposition - (|down> + i|up>)/sqrt(2)
        yield cirq.H(cirq.LineQubit(COIN_BIT))
        yield cirq.S(cirq.LineQubit(COIN_BIT))

    def add_circuit(bits):
        """Implements the addition operation on a qubit state"""
        for i in range(bits, 0, -1):
            controls = [cirq.LineQubit(v) for v in range(bits, i - 1, -1)]
            yield cirq.X(cirq.LineQubit(i - 1)).controlled_by(*controls)
            
            if i > 1:
                yield cirq.X(cirq.LineQubit(i - 1))

    def walk_step(bits):
        # Coin Flip
        yield cirq.Ry(rads=theta)(cirq.LineQubit(COIN_BIT))

        # Addition op
        yield cirq.X(cirq.LineQubit(bits))

        addition_op = cirq.Circuit(add_circuit(bits)).freeze()

        yield cirq.CircuitOperation(addition_op)
        yield cirq.X(cirq.LineQubit(bits))
        yield cirq.CircuitOperation(cirq.inverse(addition_op))

    def graph(results):
        """Simple graphing routine with matplotlib pyplot"""
        # Sort results by x-axis position
        # (Results are by default in order of decreasing frequency of measured states)
        items = [(i, j) for i, j in results.items()]
        items.sort(key=lambda x: x[0])

        # Invariant under translation - shift x-axis to 'start' position
        x_arr = [x - (1 << (num_qubits-1)) + start for x in list(zip(*items))[0]] # X-axis positions
        y_arr = [y / iterations for y in list(zip(*items))[1]] # Probability of positions

        # Plot with matplotlib
        plt.plot(x_arr, y_arr)
        plt.scatter(x_arr, y_arr)
        plt.show()


    # Create the circuit
    circuit = cirq.Circuit()
    circuit.append(init(num_qubits))

    # Add num_steps step operations
    for j in range(num_steps):
       circuit.append(walk_step(num_qubits))

    # Perform final measurement
    circuit.append(cirq.measure(*cirq.LineQubit.range(num_qubits), key='x')) # Measure final state
    
    # Run simulation many times and build histogram of data
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=iterations)
    final = result.histogram(key='x')

    # Serialize the network structure to file
    with open('circuit_line.txt', 'wt', encoding='utf-8') as f:
        f.write(str(circuit))

    # Graph the histogram results
    graph(final)


def main(argv):
    """Run the experiment"""
    if FLAGS.mode == 'classical':
        line_classical(FLAGS.iterations, FLAGS.num_steps, FLAGS.start, FLAGS.p)
    elif FLAGS.mode == 'quantum':
        line_quantum(FLAGS.iterations, FLAGS.num_steps, FLAGS.start, FLAGS.p)
    else:
        print('Invalid mode selection \"{}\" provided, must be one of \"classical\",\"quantum\"'.format(FLAGS.mode))


if __name__ == '__main__':
    """Entrypoint"""
    app.run(main)
