import numpy as np
alpha = 0.9
N = 5
# 初始化随机跳转矩阵
jump = np.full([2, 1], [[alpha], [1 - alpha]], dtype=float)
# 节点之间临接矩阵的构建
adj = np.full([N, N], [[0, 0, 1, 0, 0], [1, 0, 1, 0, 0], [
              1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]],dtype=float)
#对adj归一化
row_sums=adj.sum(axis=1)
#防止分母为0
row_sums[row_sums==0]=0.1
adj=adj/row_sums[:,np.newaxis]
#初始的pagerank值
pr=np.full([1,N],1,dtype=float)
pr_tmp=pr
delta_threshold=0.0005
for i in range(0,20):
    pr=np.dot(pr,adj)
    #对pr扩充一个维度，变成[[],[1/n,1/n]]*[alpha,1-alpha]
    pr_jump=np.full([N,2],[[0,1/N]])
    pr_jump[:,:-1]=pr.transpose()
    pr=np.dot(pr_jump,jump)
    pr=pr.transpose()
    pr=pr/pr.sum()
    delta=list(map(abs,(pr/pr_tmp)))
    delta=abs(np.max(delta)-1)
    if delta<=delta_threshold:
        print("round",i+1,pr)
        break
    else:
        pr_tmp=pr
        continue

