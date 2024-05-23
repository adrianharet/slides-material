import numpy as np
from numpy.linalg import matrix_power
from scipy import linalg
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
G.add_nodes_from([1, 2, 3])
G.add_weighted_edges_from(
    [
        (1, 1, 0.5), 
        (1, 2, 0.25), 
        (1, 3, 0.25), 
        (2, 1, 1), 
        (3, 1, 1)
    ]
)

pos = nx.spring_layout(G, seed = 7)  # positions for all nodes - seed for reproducibility

# nodes
nx.draw_networkx_nodes(
    G, 
    pos,
    node_size = 400,
    node_color="white",
    linewidths= 1,
    edgecolors="black"
    )

# edges
nx.draw_networkx_edges(
    G, 
    pos, 
    width = 1
    )

# node labels
nx.draw_networkx_labels(
    G, 
    pos, 
    font_size = 8, 
    font_family="sans-serif"
    )
# edge weight labels
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)

ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
plt.show()


c = np.array(list(nx.eigenvector_centrality_numpy(G).values()))
c = c/sum(c)
initial_beliefs = [1, 0, 0]
b = np.array([initial_beliefs]).T

W = nx.to_numpy_array(G)
L = matrix_power(W, 1000)

print("Centralities according to nx:", c)
print("Initial beliefs:\n", b)
print("Final beliefs:", np.dot(c, b))
print("Weight matrix\n", W)
print("Weight matrix to limit\n", L)
print("Limit times beliefs\n", np.dot(L, b))

eig, vl, vr = linalg.eig(W, left=True)
idx = np.argmin(np.abs(1 - eig))
s = vl[:, idx]
s /= s.sum()
print("eigenvector", s)
print("eigenvector times b", np.dot(s, b))
