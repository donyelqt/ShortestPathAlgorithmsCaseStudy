import heapq
import time
import tracemalloc
import random
import math

# Graph Algorithms
def dijkstra(graph, start):
    # Dijkstra's algorithm using a priority queue
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    prev = {node: None for node in graph}
    queue = [(0, start)]  # (distance, node)
    
    while queue:
        current_dist, node = heapq.heappop(queue)
        if current_dist > dist[node]:
            continue
        for neighbor, weight in graph[node]:
            alt = current_dist + weight
            if alt < dist[neighbor]:
                dist[neighbor] = alt
                prev[neighbor] = node
                heapq.heappush(queue, (alt, neighbor))
    
    return dist, prev

def bellman_ford(graph, start):
    # Bellman-Ford algorithm for graphs with negative weights
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    for _ in range(len(graph) - 1):
        for node in graph:
            for neighbor, weight in graph[node]:
                if dist[node] + weight < dist[neighbor]:
                    dist[neighbor] = dist[node] + weight
    return dist

def a_star(graph, start, goal, heuristic):
    # A* algorithm with Euclidean heuristic
    open_set = {start}
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start] = heuristic[start]
    
    while open_set:
        current = min(open_set, key=lambda node: f_score[node])
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        
        open_set.remove(current)
        
        for neighbor, weight in graph[current]:
            tentative_g_score = g_score[current] + weight
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic[neighbor]
                open_set.add(neighbor)
    
    return []

def floyd_warshall(graph, n):
    # Floyd-Warshall algorithm for all-pairs shortest paths
    dist = {i: {j: float('inf') for j in range(n)} for i in range(n)}
    for i in range(n):
        dist[i][i] = 0
    for i in range(n):
        for j, weight in graph[i]:
            dist[i][j] = weight
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist

def johnsons_algorithm(graph, n):
    # Johnson's algorithm using Bellman-Ford and Dijkstra
    # Step 1: Add a new vertex and connect it to all other vertices with 0-weight edges.
    graph_with_extra = graph.copy()
    for node in graph:
        graph_with_extra[node].append((n, 0))
    graph_with_extra[n] = [(i, 0) for i in range(n)]
    
    # Step 2: Run Bellman-Ford from the new vertex to get potential values.
    dist = bellman_ford(graph_with_extra, n)
    if any(dist[node] == float('inf') for node in range(n)):
        return None  # Negative weight cycle detected
    
    # Step 3: Reweight the edges and run Dijkstra from each vertex.
    reweighted_graph = {}
    for node in graph:
        reweighted_graph[node] = []
        for neighbor, weight in graph[node]:
            new_weight = weight + dist[node] - dist[neighbor]
            reweighted_graph[node].append((neighbor, new_weight))
    
    # Step 4: Run Dijkstra from each node to find shortest paths.
    shortest_paths = {}
    for node in range(n):
        shortest_paths[node] = dijkstra(reweighted_graph, node)[0]
    
    return shortest_paths

# Graph Generation
def generate_graph(num_nodes, density=0.1):
    graph = {i: [] for i in range(num_nodes)}
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < density:
                weight = random.randint(1, 10)
                graph[i].append((j, weight))
                graph[j].append((i, weight))
    return graph

# Heuristic for A* (Euclidean distance for grid-like graphs)
def euclidean_heuristic(n):
    return {i: random.randint(1, 10) for i in range(n)}

# Time and Memory Measurement
def measure_time(func, *args):
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    return result, end - start

def measure_memory(func, *args):
    tracemalloc.start()
    result = func(*args)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak

# Test scalability
def test_scalability():
    graph_sizes = [100, 200, 500]
    for size in graph_sizes:
        print(f"Testing graph with {size} nodes")
        graph = generate_graph(size, density=0.05)
        heuristic = euclidean_heuristic(size)
        
        for algo_name, algorithm in [
            ("Dijkstra", dijkstra),
            ("Bellman-Ford", bellman_ford),
            ("A*", a_star),
            ("Floyd-Warshall", floyd_warshall),
            ("Johnson's", johnsons_algorithm),
        ]:
            print(f"Running {algo_name}")
            
            # Time Measurement
            if algo_name == "A*":
                result, exec_time = measure_time(algorithm, graph, 0, size - 1, heuristic)
            else:
                result, exec_time = measure_time(algorithm, graph, 0)
            print(f"Execution Time: {exec_time:.4f} seconds")
            
            # Memory Measurement
            result, peak_memory = measure_memory(algorithm, graph, 0)
            print(f"Peak Memory Usage: {peak_memory / 10**6:.4f} MB")
            
            print("-" * 50)

# Run tests
if __name__ == "__main__":
    test_scalability()
