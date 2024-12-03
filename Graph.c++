#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <chrono>
#include <iomanip>
using namespace std;

const int INF = numeric_limits<int>::max();

// Edge representation
struct Edge {
    int src, dest, weight;
};

// Dijkstra's Algorithm
vector<int> dijkstra(int n, vector<vector<pair<int, int>>>& graph, int start) {
    vector<int> dist(n, INF);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
    dist[start] = 0;
    pq.emplace(0, start);

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) continue;

        for (auto [v, weight] : graph[u]) {
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.emplace(dist[v], v);
            }
        }
    }
    return dist;
}

// Bellman-Ford Algorithm
vector<int> bellmanFord(int n, vector<Edge>& edges, int start) {
    vector<int> dist(n, INF);
    dist[start] = 0;

    for (int i = 0; i < n - 1; ++i) {
        for (const auto& edge : edges) {
            if (dist[edge.src] != INF && dist[edge.src] + edge.weight < dist[edge.dest]) {
                dist[edge.dest] = dist[edge.src] + edge.weight;
            }
        }
    }

    // Check for negative weight cycles
    for (const auto& edge : edges) {
        if (dist[edge.src] != INF && dist[edge.src] + edge.weight < dist[edge.dest]) {
            throw runtime_error("Graph contains a negative weight cycle.");
        }
    }
    return dist;
}

// Floyd-Warshall Algorithm
vector<vector<int>> floydWarshall(int n, vector<vector<pair<int, int>>>& graph) {
    vector<vector<int>> dist(n, vector<int>(n, INF));
    for (int i = 0; i < n; ++i) dist[i][i] = 0;

    for (int u = 0; u < n; ++u) {
        for (const auto& [v, weight] : graph[u]) {
            dist[u][v] = weight;
        }
    }

    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (dist[i][k] != INF && dist[k][j] != INF) {
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                }
            }
        }
    }
    return dist;
}

// Generate Random Graph
void generateGraph(int n, double density, int maxWeight, vector<vector<pair<int, int>>>& graph, vector<Edge>& edges) {
    srand(time(0));
    for (int u = 0; u < n; ++u) {
        for (int v = 0; v < n; ++v) {
            if (u != v && (rand() % 100) < density * 100) {
                int weight = rand() % maxWeight + 1;
                graph[u].emplace_back(v, weight);
                edges.push_back({u, v, weight});
            }
        }
    }
}

int main() {
    int n = 100; // Number of nodes
    double density = 0.3; // Graph density
    int maxWeight = 10; // Maximum edge weight
    vector<vector<pair<int, int>>> graph(n);
    vector<Edge> edges;

    generateGraph(n, density, maxWeight, graph, edges);

    // Measure time for Dijkstra's Algorithm
    auto start = chrono::high_resolution_clock::now();
    auto dijkstraDistances = dijkstra(n, graph, 0);
    auto end = chrono::high_resolution_clock::now();
    cout << "Dijkstra's Algorithm Runtime: " 
         << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

    // Measure time for Bellman-Ford Algorithm
    start = chrono::high_resolution_clock::now();
    try {
        auto bellmanFordDistances = bellmanFord(n, edges, 0);
    } catch (runtime_error& e) {
        cout << e.what() << endl;
    }
    end = chrono::high_resolution_clock::now();
    cout << "Bellman-Ford Algorithm Runtime: " 
         << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

    // Measure time for Floyd-Warshall Algorithm
    start = chrono::high_resolution_clock::now();
    auto floydWarshallDistances = floydWarshall(n, graph);
    end = chrono::high_resolution_clock::now();
    cout << "Floyd-Warshall Algorithm Runtime: " 
         << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

    return 0;
}
