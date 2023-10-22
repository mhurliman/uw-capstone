import cirq
import math

bits = 5

psi = cirq.LineQubit.range(bits)

for i in range(bits-1, 0):
    cirq.H(psi[i])
    factor = math.pi / 2
    for j in range(i):
        cirq.Gate.controlled()
        cirq.ZPowGate(exponent=factor)
        factor /= 2