import networkx as nx

def solve_max_flow(adj_matrix, source_indices, sink_indices):
    """
    Solves the Maximum Flow problem given an adjacency matrix representing capacities,
    a list of source node indices, and a list of sink node indices.
    
    Args:
        adj_matrix (list of list of int): Adjacency matrix where adj_matrix[i][j] is the capacity of the edge from i to j.
        source_indices (list of int): List of integer indices corresponding to source nodes.
        sink_indices (list of int): List of integer indices corresponding to sink nodes.
        
    Returns:
        int: The maximum flow value through the network.
    """
    # Create a directed graph from the adjacency matrix
    G = nx.DiGraph()
    num_nodes = len(adj_matrix)
    
    # Add nodes
    for i in range(num_nodes):
        G.add_node(i)
        
    # Add edges with capacities
    for i in range(num_nodes):
        for j in range(num_nodes):
            capacity = adj_matrix[i][j]
            if capacity > 0:
                G.add_edge(i, j, capacity=capacity)
    
    # Create a super-source and super-sink to handle multiple sources and sinks
    super_source = 'super_source'
    super_sink = 'super_sink'
    
    G.add_node(super_source)
    G.add_node(super_sink)
    
    # Connect super-source to all actual sources with infinite capacity
    for s_idx in source_indices:
        G.add_edge(super_source, s_idx, capacity=float('inf'))
        
    # Connect all actual sinks to super-sink with infinite capacity
    for t_idx in sink_indices:
        G.add_edge(t_idx, super_sink, capacity=float('inf'))
        
    # Calculate max flow using Edmonds-Karp algorithm (or preflow-push)
    # NetworkX's maximum_flow defaults to preflow-push which is generally faster
    flow_value, flow_dict = nx.maximum_flow(G, super_source, super_sink)
    
    return flow_value

# Example usage:
if __name__ == "__main__":
    # Example adjacency matrix (capacities)
    # Nodes: 0, 1, 2, 3
    # 0 -> 1 (cap 3), 0 -> 2 (cap 2)
    # 1 -> 2 (cap 5), 1 -> 3 (cap 2)
    # 2 -> 3 (cap 3)
    capacities = [
        [0, 3, 2, 0],
        [0, 0, 5, 2],
        [0, 0, 0, 3],
        [0, 0, 0, 0]
    ]
    
    sources = [0]
    sinks = [3]
    
    max_flow = solve_max_flow(capacities, sources, sinks)
    print(f"Maximum Flow: {max_flow}")
