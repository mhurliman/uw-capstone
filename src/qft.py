import cirq
import math

def QFT(bits):
    for i in range(len(bits)):
        # Each qubit begins in superposition
        yield cirq.H(bits[i])

        # Encode state into phase of qubit using controlled Z rotation gates
        exp = 0.5
        for j in range(i + 1, len(bits)):
            yield cirq.Rz(rads=math.pi*exp)(bits[i]).controlled_by(bits[j])
            exp *= 0.5

def iQFT(bits):
    yield cirq.inverse(QFT(bits))
