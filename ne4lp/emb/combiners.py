import numpy as np

summ = np.add
diff = np.subtract
hadamard = np.multiply
multiply = hadamard


def squareddiff(a, b):
    return np.power(a - b, 2)


def absdiff(a, b):
    return np.abs(a - b)


def avg(a, b):
    return (a + b) / 2
