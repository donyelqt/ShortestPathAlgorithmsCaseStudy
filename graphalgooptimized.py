import time
import tracemalloc
from queue import PriorityQueue
from collections import defaultdict
import random
import math
import matplotlib.pyplot as plt
import numpy as np

# Graph generation utilities
def generate_random_graph(num_nodes, density=0.3, max_weight=10):
    graph = defaultdict(list)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and random.random() < density:
                weight = random.randint(1, max_weight)
                graph[i].append((j, weight))
    return graph

def generate_complete_graph(num_nodes, max_weight=10):
    graph = defaultdict(list)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                weight = random.randint(1, max_weight)
                graph[i].append((j, weight))
    return graph

def generate_sparse_graph(num_nodes, max_weight=10):
    return generate_random_graph(num_nodes, density=0.1, max_weight=max_weight)

def generate_dense_graph(num_nodes, max_weight=10):
    return generate_random_graph(num_nodes, density=0.8, max_weight=max_weight)

def generate_cyclic_graph(num_nodes):
    graph = defaultdict(list)
    for i in range(num_nodes):
        graph[i].append(((i + 1) % num_nodes, 1))  # Connect to the next node
    return graph

# Algorithms
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = PriorityQueue()
    pq.put((0, start))

    while not pq.empty():
        current_distance, current_node = pq.get()
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                pq.put((distance, neighbor))
    return distances

def bellman_ford(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    for _ in range(len(graph) - 1):
        for node in graph:
            for neighbor, weight in graph[node]:
                if distances[node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[node] + weight
    for node in graph:
        for neighbor, weight in graph[node]:
            if distances[node] + weight < distances[neighbor]:
                raise ValueError("Graph contains a negative weight cycle")
    return distances

def floyd_warshall(graph, num_nodes):
    dist = [[float('inf')] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        dist[i][i] = 0
    for u in graph:
        for v, w in graph[u]:
            dist[u][v] = w
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist

def a_star(graph, start, goal, heuristic):
    open_set = PriorityQueue()
    open_set.put((0, start))
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    while not open_set.empty():
        _, current = open_set.get()
        if current == goal:
            return g_score
        for neighbor, weight in graph[current]:
            tentative_g_score = g_score[current] + weight
            if tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                open_set.put((g_score[neighbor] + heuristic[neighbor], neighbor))
    return g_score

def johnsons_algorithm(graph, num_nodes):
    new_graph = {i: graph[i] + [(num_nodes, 0)] for i in graph}
    new_graph[num_nodes] = [(i, 0) for i in range(num_nodes)]
    h = bellman_ford(new_graph, num_nodes)
    reweighted_graph = defaultdict(list)
    
    # Ensure all nodes are included in the reweighted graph
    for u in range(num_nodes):
        for v, w in graph[u]:
            reweighted_graph[u].append((v, w + h[u] - h[v]))
        # Ensure nodes with no outgoing edges are still included
        if u not in reweighted_graph:
            reweighted_graph[u] = []
    
    all_pairs_dist = []
    for node in range(num_nodes):
        all_pairs_dist.append(dijkstra(reweighted_graph, node))
    return all_pairs_dist

def euclidean_heuristic(num_nodes):
    heuristic = {}
    # Use a simple heuristic based on node indices
    for node in range(num_nodes):
        if node == num_nodes - 1:  # Goal node
            heuristic[node] = 0
        else:
            heuristic[node] = abs(node - (num_nodes - 1))  # Simple heuristic based on node index
    return heuristic

# Performance Analysis
def analyze_performance(graph, start, goal):
    results = {}

    # Dijkstra
    start_time = time.perf_counter()
    dijkstra_result = dijkstra(graph, start)
    results['Dijkstra'] = (time.perf_counter() - start_time, dijkstra_result[goal])

    # Bellman-Ford
    start_time = time.perf_counter()
    bellman_ford_result = bellman_ford(graph, start)
    results['Bellman-Ford'] = (time.perf_counter() - start_time, bellman_ford_result[goal])

    # A*
    heuristic = euclidean_heuristic(len(graph))
    start_time = time.perf_counter()
    a_star_result = a_star(graph, start, goal, heuristic)
    results['A*'] = (time.perf_counter() - start_time, a_star_result[goal])

    # Floyd-Warshall
    start_time = time.perf_counter()
    floyd_result = floyd_warshall(graph, len(graph))
    results['Floyd-Warshall'] = (time.perf_counter() - start_time, floyd_result[start][goal])

    # Johnson's Algorithm
    start_time = time.perf_counter()
    johnson_result = johnsons_algorithm(graph, len(graph))
    results['Johnson'] = (time.perf_counter() - start_time, johnson_result[start][goal])

    return results

# Testing and Validation
def validate_algorithms():
    # Use a fixed graph for validation
    graph = {
        0: [(1, 4), (2, 1)],
        1: [(3, 1)],
        2: [(1, 2), (3, 5)],
        3: []
    }
    expected = {
        "Dijkstra": {0: 0, 1: 3, 2: 1, 3: 4},
        "Bellman-Ford": {0: 0, 1: 3, 2: 1, 3: 4}
    }
    for algo, func in [("Dijkstra", dijkstra), ("Bellman-Ford", bellman_ford)]:
        output = func(graph, 0)  # Always use the fixed graph
        assert output == expected[algo], f"{algo} failed validation!"

    # Validate A*, Floyd-Warshall, and Johnson's algorithms
    heuristic = euclidean_heuristic(len(graph))
    a_star_result = a_star(graph, 0, 3, heuristic)
    assert a_star_result[3] == 4, "A* failed validation!"

    floyd_result = floyd_warshall(graph, len(graph))
    assert floyd_result[0][3] == 4, "Floyd-Warshall failed validation!"

    johnson_result = johnsons_algorithm(graph, len(graph))
    assert johnson_result[0][3] == 4, "Johnson's algorithm failed validation!"

# Scalability Testing
def test_scalability(iterations=2):
    sizes = [100, 200, 500]  # Removed 1000 for performance
    times = {algo: [] for algo in ["Dijkstra", "Bellman-Ford", "A*", "Floyd-Warshall", "Johnson's"]}
    
    for size in sizes:
        for _ in range(iterations):
            graph = generate_random_graph(size)
            results = analyze_performance(graph, 0, size - 1)
            for algo in results:
                times[algo].append(results[algo][0])  # Append the time taken
    
    # Average the times
    avg_times = {algo: np.mean(times[algo]) for algo in times}
    
    for algo in avg_times:
        plt.plot(sizes, avg_times[algo], label=algo)
    plt.xlabel("Number of Nodes")
    plt.ylabel("Average Time (seconds)")
    plt.legend()
    plt.title("Algorithm Scalability")
    plt.show()

# Memory Usage Measurement
def measure_memory(func, *args):
    tracemalloc.start()
    func(*args)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return current, peak

def analyze_memory_usage(iterations=2):
    sizes = [100, 200, 500]  # Removed 1000 for performance
    memory_usage = {algo: [] for algo in ["Dijkstra", "Bellman-Ford", "A*", "Floyd-Warshall", "Johnson's"]}
    
    for size in sizes:
        for _ in range(iterations):
            graph = generate_random_graph(size)
            heuristic = euclidean_heuristic(size)
            for algo, func in [
                ("Dijkstra", lambda: dijkstra(graph, 0)),
                ("Bellman-Ford", lambda: bellman_ford(graph, 0)),
                ("A*", lambda: a_star(graph, 0, size - 1, heuristic)),
                ("Floyd-Warshall", lambda: floyd_warshall(graph, size)),
                ("Johnson's", lambda: johnsons_algorithm(graph, size))
            ]:
                current, peak = measure_memory(func)
                memory_usage[algo].append(peak)
    
    # Average the memory usage
    avg_memory = {algo: np.mean(memory_usage[algo]) for algo in memory_usage}
    
    for algo in avg_memory:
        plt.plot(sizes, avg_memory[algo], label=algo)
    plt.xlabel("Number of Nodes")
    plt.ylabel("Average Memory Usage (bytes)")
    plt.legend()
    plt.title("Algorithm Memory Usage")
    plt.show()

if __name__ == "__main__":
    num_nodes = 100
    graph = generate_random_graph(num_nodes)
    validate_algorithms()  # Validate using the fixed graph
    test_scalability(iterations=2)  # Test time complexity with multiple iterations
    analyze_memory_usage(iterations=2)  # Test space complexity with multiple iterations