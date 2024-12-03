import time
import sys
import tracemalloc
from queue import PriorityQueue
from collections import defaultdict
import random
import math

# Graph generation utility
def generate_graph(num_nodes, density=0.3, max_weight=10):
    graph = defaultdict(list)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and random.random() < density:
                weight = random.randint(1, max_weight)
                graph[i].append((j, weight))
    return graph

# Dijkstra's Algorithm
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

# Bellman-Ford Algorithm with Negative Cycle Detection
def bellman_ford(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    for _ in range(len(graph) - 1):
        for node in graph:
            for neighbor, weight in graph[node]:
                if distances[node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[node] + weight
    # Check for negative weight cycles
    for node in graph:
        for neighbor, weight in graph[node]:
            if distances[node] + weight < distances[neighbor]:
                raise ValueError("Graph contains a negative weight cycle")
    return distances

# Floyd-Warshall Algorithm
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

# A* Search Algorithm with Euclidean Heuristic
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

# Johnson’s Algorithm
def johnsons_algorithm(graph, num_nodes):
    new_graph = {i: graph[i] + [(num_nodes, 0)] for i in graph}
    new_graph[num_nodes] = [(i, 0) for i in range(num_nodes)]
    h = bellman_ford(new_graph, num_nodes)
    reweighted_graph = defaultdict(list)
    for u in graph:
        for v, w in graph[u]:
            reweighted_graph[u].append((v, w + h[u] - h[v]))
    all_pairs_dist = []
    for node in range(num_nodes):
        all_pairs_dist.append(dijkstra(reweighted_graph, node))
    return all_pairs_dist

# Heuristic for A* (Euclidean Distance)
def euclidean_heuristic(num_nodes):
    heuristic = {}
    positions = {i: (random.randint(0, 100), random.randint(0, 100)) for i in range(num_nodes)}
    for node in positions:
        heuristic[node] = math.sqrt((positions[node][0] - positions[num_nodes-1][0])**2 +
                                    (positions[node][1] - positions[num_nodes-1][1])**2)
    return heuristic

# Time, Space, and Optimality Testing
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
    return results

if __name__ == "__main__":
    num_nodes = 100
    graph = generate_graph(num_nodes)
    results = test_algorithms(graph, num_nodes)
    for algo, metrics in results.items():
        print(f"\n{algo} Results:")
        print(f"Time: {metrics['Time']} seconds")
        print(f"Memory: {metrics['Memory']} bytes")
        print(f"Optimality: {metrics['Optimality']}")
