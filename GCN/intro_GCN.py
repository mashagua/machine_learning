import numpy as np
A = np.matrix([[0, 1, 0, 0],
               [0, 0, 1, 1],
               [0, 1, 0, 0],
               [1, 0, 1, 0]], dtype=float)

X = np.matrix([[i, -i] for i in range(A.shape[0])], dtype=float)
print(X)
