import numpy as np
# A = np.matrix([[0, 1, 0, 0],
#                [0, 0, 1, 1],
#                [0, 1, 0, 0],
#                [1, 0, 1, 0]], dtype=float)
#
# X = np.matrix([[i, -i] for i in range(A.shape[0])], dtype=float)
# print(X)
# print(A*X)
# I=np.matrix(np.eye(A.shape[0]))
# A_HAT=A+I
# print(A_HAT*X)
# #0是列，1是行
# D=np.array(np.sum(A,axis=0))[0]
# D=np.matrix(np.diag(D))

# print(D_HAT)
# print(D**-1*A*X)
#
# W=np.matrix([[1,-1],[-1,1]])
# relu函数作用
# print(np.maximum(D_HAT**-1*A_HAT*X*W,0))
from networkx import karate_club_graph, to_numpy_matrix
zkc = karate_club_graph()
order = sorted(list(zkc.nodes()))
A = to_numpy_matrix(zkc, nodelist=order)
I = np.eye(zkc.number_of_nodes())
A_HAT = A + I
D_HAT = np.array(np.sum(A_HAT, axis=0))[0]
D_HAT = np.matrix(np.diag(D_HAT))
w_1 = np.random.normal(loc=0, scale=1, size=(zkc.number_of_nodes(), 4))
w_2 = np.random.normal(loc=0, size=(w_1.shape[1], 2))


def gcn_layer(A_HAT, D_HAT, X, W):
    return np.maximum(D_HAT**-1 * A_HAT * X * W, 0)


H_1 = gcn_layer(A_HAT, D_HAT, I, w_1)
H_2 = gcn_layer(A_HAT, D_HAT, H_1, w_2)
output = H_2
feature_representations = {node: np.array(
    output)[node] for node in zkc.nodes()}
print(feature_representations)
