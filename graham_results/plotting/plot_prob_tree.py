# ---- This is a function which takes the current status ---
# ---- of a wavefunction and creates a binary tree which ---
# ---- tracks the probabilities of each configuration ------

import os
import matplotlib.pyplot as plt
import networkx as nx

from networkx.drawing.nx_agraph import write_dot, graphviz_layout


def build_tree(N, wavefunc, filename):
    G = nx.balanced_tree(2, N)  # define empty binary tree

    probs = abs(wavefunc) ** 2  # get probabilities from wavefunction

    mapping_dict = {}

    # assign leaves to probabilities of basis states
    for idx in range(len(probs)):
        bfs_idx = 2 ** N  # since nodes are assigned numbers breadth-first
        mapping_dict[idx + bfs_idx - 1] = round(probs[idx], 3)

    # map other nodes to sum of probabilities of children
    for layer in range(N - 1, 0, -1):
        for idx in range(2 ** layer):
            bfs_idx = 2 ** layer  # since nodes are assigned numbers breadth-first
            mapping_dict[bfs_idx + idx - 1] = round(
                mapping_dict[2 * bfs_idx + 2 * idx - 1]
                + mapping_dict[2 * bfs_idx + 2 * idx],
                3,
            )

    mapping_dict[0] = round(mapping_dict[1] + mapping_dict[2], 3)  # add root node

    # label edges with spins
    edge_labels_dict = {}

    spin = 0
    for edge in G.edges():
        edge_labels_dict[edge] = spin
        spin = (spin + 1) % 2  # alternate spins since edges are ordered

    # make directed binary tree graph
    write_dot(G, "temp.dot")

    plt.subplots(figsize=(20, 5))
    pos = graphviz_layout(G, prog="dot")
    nx.draw(G, pos, with_labels=False, arrows=True, node_size=1000)
    nx.draw_networkx_labels(G, pos, mapping_dict)
    nx.draw_networkx_edge_labels(G, pos, edge_labels_dict)

    plt.tight_layout()
    plt.savefig("{0}.png".format(filename))

    os.remove("temp.dot")
