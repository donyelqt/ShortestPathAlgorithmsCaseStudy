import random
import time
import tracemalloc
from collections import defaultdict
from memory_profiler import profile

# Example algorithms (Dijkstra, Bellman-Ford, Floyd-Warshall, A*, etc.)
def dijkstra(graph, start):
    # Dijkstra's algorithm implementation here
    pass

def bellman_ford(graph, start):
    # Bellman-Ford algorithm implementation here
    pass

def floyd_warshall(graph, num_nodes):
    # Floyd-Warshall algorithm implementation here
    pass

def a_star(graph, start, goal, heuristic):
    # A* algorithm implementation here
    pass

def johnsons_algorithm(graph, num_nodes):
    # Johnson's algorithm implementation here
    pass

# Euclidean heuristic function for A* (just an example)
def euclidean_heuristic(num_nodes):
    return lambda u, v: abs(u - v)

# Graph generation functions
def generate_graph(num_nodes, density=0.3, max_weight=10):
    graph = defaultdict(list)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and random.random() < density:
                weight = random.randint(1, max_weight)
                graph[i].append((j, weight))
    return graph

def generate_graph_with_negative_weights(num_nodes, density=0.3, max_weight=10):
    graph = defaultdict(list)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and random.random() < density:
                weight = random.randint(-max_weight, max_weight)
                graph[i].append((j, weight))
    return graph

# Ground truth comparison using Floyd-Warshall
def compare_with_ground_truth(graph, num_nodes):
    true_distances = floyd_warshall(graph, num_nodes)
    dijkstra_result = dijkstra(graph, 0)  # Assuming we start from node 0
    for node in range(num_nodes):
        assert dijkstra_result[node] == true_distances[0][node], f"Mismatch at node {node}"

# Profiling decorator
@profile
def test_algorithm_memory(graph, num_nodes, algorithm):
    result = algorithm(graph, num_nodes)
    return result

# Algorithm performance test
def test_algorithms(graph, num_nodes):
    heuristic = euclidean_heuristic(num_nodes)
    algorithms = {
        "Dijkstra": lambda: dijkstra(graph, 0),
        "Bellman-Ford": lambda: bellman_ford(graph, 0),
        "Floyd-Warshall": lambda: floyd_warshall(graph, num_nodes),
        "A*": lambda: a_star(graph, 0, num_nodes - 1, heuristic),
        "Johnson's": lambda: johnsons_algorithm(graph, num_nodes)
    }

    results = {}
    for name, algo in algorithms.items():
        print(f"Running {name}...")
        start_time = time.time()
        tracemalloc.start()
        try:
            output = algo()
        except ValueError as e:
            output = str(e)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()
        
        results[name] = {
            "Time": end_time - start_time,
            "Memory": peak,
            "Optimality": output
        }

    # Compare with ground truth for small graphs
    if num_nodes <= 10:
        compare_with_ground_truth(graph, num_nodes)

    return results

if __name__ == "__main__":
    # Test with various graph types and configurations
    num_nodes = 100
    graph_sparse = generate_graph(num_nodes, density=0.1)
    graph_dense = generate_graph(num_nodes, density=0.9)
    graph_with_negative_weights = generate_graph_with_negative_weights(num_nodes)

    # Run tests on different graph types
    results_sparse = test_algorithms(graph_sparse, num_nodes)
    results_dense = test_algorithms(graph_dense, num_nodes)
    results_negative_weights = test_algorithms(graph_with_negative_weights, num_nodes)

    # Output results
    print("\nSparse Graph Results:")
    for algo, metrics in results_sparse.items():
        print(f"{algo}: Time = {metrics['Time']}s, Memory = {metrics['Memory']} bytes")

    print("\nDense Graph Results:")
    for algo, metrics in results_dense.items():
        print(f"{algo}: Time = {metrics['Time']}s, Memory = {metrics['Memory']} bytes")

    print("\nNegative Weight Graph Results:")
    for algo, metrics in results_negative_weights.items():
        print(f"{algo}: Time = {metrics['Time']}s, Memory = {metrics['Memory']} bytes")
