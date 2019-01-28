import numpy as np
from random import choices

def power(sample1, sample2, reps, size, alpha):

    for i in range(reps):
        gen1 = choices(sample1, k=size)
        gen2 = choices(sample2, k=size)

        