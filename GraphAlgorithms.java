import java.util.*;

public class GraphAlgorithms {

    static class Edge {
        int src, dest, weight;
        Edge(int src, int dest, int weight) {
            this.src = src;
            this.dest = dest;
            this.weight = weight;
        }
    }

    // Dijkstra's Algorithm
    public static Map<Integer, Integer> dijkstra(Map<Integer, List<Edge>> graph, int start, int numNodes) {
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
        Map<Integer, Integer> distances = new HashMap<>();
        for (int i = 0; i < numNodes; i++) distances.put(i, Integer.MAX_VALUE);
        distances.put(start, 0);
        pq.add(new int[]{start, 0});
        
        while (!pq.isEmpty()) {
            int[] current = pq.poll();
            int node = current[0], distance = current[1];
            if (distance > distances.get(node)) continue;
            for (Edge edge : graph.get(node)) {
                int newDist = distance + edge.weight;
                if (newDist < distances.get(edge.dest)) {
                    distances.put(edge.dest, newDist);
                    pq.add(new int[]{edge.dest, newDist});
                }
            }
        }
        return distances;
    }

    // Bellman-Ford Algorithm
    public static Map<Integer, Integer> bellmanFord(List<Edge> edges, int start, int numNodes) {
        Map<Integer, Integer> distances = new HashMap<>();
        for (int i = 0; i < numNodes; i++) distances.put(i, Integer.MAX_VALUE);
        distances.put(start, 0);

        for (int i = 0; i < numNodes - 1; i++) {
            for (Edge edge : edges) {
                if (distances.get(edge.src) != Integer.MAX_VALUE &&
                        distances.get(edge.src) + edge.weight < distances.get(edge.dest)) {
                    distances.put(edge.dest, distances.get(edge.src) + edge.weight);
                }
            }
        }

        // Check for negative cycles
        for (Edge edge : edges) {
            if (distances.get(edge.src) != Integer.MAX_VALUE &&
                    distances.get(edge.src) + edge.weight < distances.get(edge.dest)) {
                throw new IllegalArgumentException("Graph contains a negative weight cycle");
            }
        }
        return distances;
    }

    // Floyd-Warshall Algorithm
    public static int[][] floydWarshall(Map<Integer, List<Edge>> graph, int numNodes) {
        int[][] dist = new int[numNodes][numNodes];
        for (int i = 0; i < numNodes; i++) {
            Arrays.fill(dist[i], Integer.MAX_VALUE);
            dist[i][i] = 0;
        }
        for (int u : graph.keySet()) {
            for (Edge edge : graph.get(u)) {
                dist[u][edge.dest] = edge.weight;
            }
        }
        for (int k = 0; k < numNodes; k++) {
            for (int i = 0; i < numNodes; i++) {
                for (int j = 0; j < numNodes; j++) {
                    if (dist[i][k] != Integer.MAX_VALUE && dist[k][j] != Integer.MAX_VALUE) {
                        dist[i][j] = Math.min(dist[i][j], dist[i][k] + dist[k][j]);
                    }
                }
            }
        }
        return dist;
    }

    public static void main(String[] args) {
        int numNodes = 100;
        Map<Integer, List<Edge>> graph = generateGraph(numNodes, 0.3, 10);

        long startTime, endTime;

        // Dijkstra's Algorithm
        startTime = System.nanoTime();
        Map<Integer, Integer> dijkstraDistances = dijkstra(graph, 0, numNodes);
        endTime = System.nanoTime();
        System.out.println("Dijkstra Time: " + (endTime - startTime) / 1e6 + " ms");

        // Bellman-Ford Algorithm
        List<Edge> edges = convertGraphToEdgeList(graph);
        startTime = System.nanoTime();
        try {
            Map<Integer, Integer> bellmanFordDistances = bellmanFord(edges, 0, numNodes);
        } catch (IllegalArgumentException e) {
            System.out.println(e.getMessage());
        }
        endTime = System.nanoTime();
        System.out.println("Bellman-Ford Time: " + (endTime - startTime) / 1e6 + " ms");

        // Floyd-Warshall Algorithm
        startTime = System.nanoTime();
        int[][] floydDistances = floydWarshall(graph, numNodes);
        endTime = System.nanoTime();
        System.out.println("Floyd-Warshall Time: " + (endTime - startTime) / 1e6 + " ms");
    }

    private static Map<Integer, List<Edge>> generateGraph(int numNodes, double density, int maxWeight) {
        Map<Integer, List<Edge>> graph = new HashMap<>();
        Random random = new Random();
        for (int i = 0; i < numNodes; i++) {
            graph.put(i, new ArrayList<>());
        }
        for (int i = 0; i < numNodes; i++) {
            for (int j = 0; j < numNodes; j++) {
                if (i != j && random.nextDouble() < density) {
                    graph.get(i).add(new Edge(i, j, random.nextInt(maxWeight) + 1));
                }
            }
        }
        return graph;
    }

    private static List<Edge> convertGraphToEdgeList(Map<Integer, List<Edge>> graph) {
        List<Edge> edges = new ArrayList<>();
        for (int u : graph.keySet()) {
            edges.addAll(graph.get(u));
        }
        return edges;
    }
}
