
def prune_graph(G, min_degree):
    nodes_to_remove = [node for node, degree in dict(G.degree()).items() if degree <= min_degree]
    G_pruned = G.copy()
    G_pruned.remove_nodes_from(nodes_to_remove)
    return G_pruned