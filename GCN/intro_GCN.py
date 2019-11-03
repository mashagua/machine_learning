import numpy as np
A = np.matrix([[0, 1, 0, 0],
               [0, 0, 1, 1],
               [0, 1, 0, 0],
               [1, 0, 1, 0]], dtype=float)

X = np.matrix([[i, -i] for i in range(A.shape[0])], dtype=float)
print(X)
print(A*X)
I=np.matrix(np.eye(A.shape[0]))
A_HAT=A+I
print(A_HAT*X)
D=np.array(np.sum(A,axis=0))[0]
D=np.matrix(np.diag(D))

