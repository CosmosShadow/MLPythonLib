# coding: utf-8
from igraph import *

g = Graph()

# 添加
g.add_vertices(5)
g.add_edges([(0, 1), (1, 2)])

# 删除
# g.delete_vertices([0])
# g.delete_edges()

# 根据顶点获取边
print g.get_eid(1, 2)

print g

# 输出summary
summary(g)

# 生成树，分3叉
g = Graph.Tree(127, 3)
summary(g)

# 获取所有的边
print g.get_edgelist()

# 随机生成: 在单位正方形中随机落下点，距离小于0.2的连接
g = Graph.GRG(100, 0.2)

layout = g.layout("kk")
plot(g, "1.pdf", layout = layout)