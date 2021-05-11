import numpy as np

A = np.arange(5)
np.random.shuffle(A)

A = [2, 1, 3, 3, 3, 3, 3, 2, ]
A = list(A)
B = sorted(A)

print(A)
print(B)

sum_ = 0
groups = 0

for l in A:
    k = A.index(l) - B.index(l)

    sum_ += k
    if sum_ == 0:
        groups += 1

print(groups)

