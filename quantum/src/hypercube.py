from absl import (app, flags)
import cirq
import math
import matplotlib.pyplot as plt

# Command line flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('iterations', 1000, 'Number of experiment iterations to run to garner a meaningful sample of the distribution.')
flags.DEFINE_list('marked_nodes', ['13', '15'], 'Integral indices of the marked nodes of the hypercube (0 - 15).')


def hypercube(iterations, marked_nodes):
    """
    Performs a Grover-coined quantum walk on the 4D hypercube finding marked states, Quantum Walk 'Search' algorithm.
    Given a set of marked hypercube vertices the quantum walk will 'find' them through graph traversal

    :param iteratons: The number of experiment iterations to perform
    :param marked_nodes: The list of marked node indices in which to find (0 - 15)
    """

    def shift_operator(qubits):
        """The 'Shift' operation of the walk - performs a single step of the walk"""
        for i in range(4):
            yield cirq.X(qubits[4])
            if i % 2 == 0:
                yield cirq.X(qubits[5])

            yield cirq.X(qubits[i]).controlled_by(*qubits[4:6])

    def one_step(qubits):
        """One full step of the walk - includes the Shift and Coin operators"""
        c = cirq.Circuit()

        # Grover coin operator
        c.append(cirq.H.on_each(*qubits))
        c.append(cirq.Z.on_each(*qubits))
        c.append(cirq.Z(qubits[5]).controlled_by(qubits[4]))
        c.append(cirq.H.on_each(*qubits))

        # Shift operator
        c.append(shift_operator(qubits))

        return c

    def phase_oracle(qubits, nodes):
        """Modifies the phase of the specified marked states"""
        c = cirq.Circuit()

        for s in nodes:
            # Create an X gate for each '0' bit (endianness makes this a bit awkward)
            X_gates = [cirq.X(qubits[i]) for i in range(4) if ((s >> (3 - i)) & 0x1) == 0]

            # Mark the specified state
            c.append(X_gates + [cirq.H(qubits[3])])
            c.append(cirq.X(qubits[3]).controlled_by(*qubits[0:3]))
            c.append(X_gates + [cirq.H(qubits[3])])

        return c

    def mark_auxiliary(qubits):
        """Used for phase estimation - marks an auxiliary qubit if a 4-bit 'theta' parameter != 0"""
        c = cirq.Circuit()
        
        c.append(cirq.X.on_each(*qubits))
        c.append(cirq.X(qubits[4]).controlled_by(*qubits[0:4]))
        c.append(cirq.Z(qubits[4]))
        c.append(cirq.X(qubits[4]).controlled_by(*qubits[0:4]))
        c.append(cirq.X.on_each(*qubits))

        return c

    def phase_estimation(qubits):
        """Performs phase estimation of the quantum states"""
        c = cirq.Circuit()

        c.append(cirq.H.on_each(*qubits[0:4])) 
        
        for i in range(0, 4):
            stop = 2**i
            for j in range(0, stop):
                one_step_op = cirq.CircuitOperation(one_step(qubits[4:10]).freeze())
                c.append(one_step_op.controlled_by(qubits[i]))

        c.append(cirq.inverse(cirq.qft(*qubits[0:4])))
        c.append(mark_auxiliary(qubits[0:4] + [qubits[10]]))
        c.append(cirq.qft(*qubits[0:4]))

        for i in range(3, -1, -1):
            stop = 2**i
            for j in range(0, stop):
                one_step_op_inv = cirq.CircuitOperation(cirq.inverse(one_step(qubits[4:10])).freeze())
                c.append(one_step_op_inv.controlled_by(qubits[i]))

        c.append(cirq.H.on_each(*qubits[0:4]), cirq.InsertStrategy.NEW)

        return c

    def graph(results):
        """Simple graphing routine with matplotlib pyplot"""
        # Sort results by x-axis position
        # (Results are by default in order of decreasing frequency of measured states)
        items = [(i, j) for i, j in results.items()]
        items.sort(key=lambda x: x[0])

        x_arr = list(zip(*items))[0] # Node IDs
        y_arr = [y / iterations for y in list(zip(*items))[1]] # Probability of nodes

        # Plot with matplotlib
        plt.bar(x_arr, y_arr)
        plt.show()


    # Instaniate our qubits
    theta = cirq.LineQubit.range(0, 4)
    node = cirq.LineQubit.range(4, 8)
    coin = cirq.LineQubit.range(8, 10)
    auxiliary = cirq.LineQubit(10)

    # Marked states must be valid (0 -> 15) & unique
    marked_nodes = set(max(0, min(x, 15)) for x in marked_nodes)

    # Bail if no valid marked states
    if len(marked_nodes) == 0:
        print('No marked states were specified')
        return

    # Start building the circuit
    c = cirq.Circuit()
    c.append(cirq.H.on_each(*(node + coin)))

    # We must iterate the algorithm 1/sqrt(|M|/N) times
    runs = math.floor(1.0 / math.sqrt(len(marked_nodes) / 16.0))
    for i in range(runs):
        c.append(phase_oracle(node, marked_nodes).freeze())
        c.append(phase_estimation(theta + node + coin + [auxiliary]).freeze())

    # Perform final measurement
    c.append(cirq.measure(*node, key='node'))

    # Run the quantum simulator
    sim = cirq.Simulator()
    result = sim.run(c, repetitions=iterations)

    # Export the circuit structure to file
    with open('circuit_hypercube.txt', 'wt', encoding='utf-8') as f:
        f.write(str(c))

    # Gather and graph the results
    final = result.histogram(key='node')
    graph(final)


def main(argv):
    """Run the experiment"""
    FLAGS.marked_nodes = [int(n) for n in FLAGS.marked_nodes] # Santitize the node list input
    hypercube(FLAGS.iterations, FLAGS.marked_nodes)


if __name__ == '__main__':
    """Entrypoint"""
    app.run(main)
