import numpy as np 
import itertools

# ==============================================
# Task 1: Graph Representation and Data Structures
# ==============================================

# Adjacency matrix representation of the graph
# Cities are labeled from 1 to 7
# Distance from city i to city j is stored in matrix[i-1][j-1]

# Initialize the matrix with infinity (no direct connection)
INF = float('inf')
graph = [
    [0, 10, 15, 20, INF, INF, INF],  # City 1
    [10, 0, 35, 25, INF, INF, INF],  # City 2
    [15, 35, 0, 30, 12, INF, INF],   # City 3
    [20, 25, 30, 0, 18, 8, INF],     # City 4
    [INF, INF, 12, 18, 0, 22, 14],   # City 5
    [INF, INF, INF, 8, 22, 0, 24],   # City 6
    [INF, INF, INF, INF, 14, 24, 0]  # City 7
]

# ==============================================
# Task 2: Classical TSP Solution (Dynamic Programming)
# ==============================================

def tsp_dp(graph):
    n = len(graph)
    # dp[mask][i] : minimum cost to visit all cities in mask ending at city i
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Start at city 1 (mask = 1, last city = 0)

    # Iterate over all subsets of cities
    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)):
                continue  # Skip if city u is not in the subset
            for v in range(n):
                if mask & (1 << v):
                    continue  # Skip if city v is already in the subset
                dp[mask | (1 << v)][v] = min(dp[mask | (1 << v)][v], dp[mask][u] + graph[u][v])

    # Return to the starting city (city 1)
    final_mask = (1 << n) - 1
    min_distance = min(dp[final_mask][u] + graph[u][0] for u in range(n))

    # Reconstruct the route
    route = []
    mask = final_mask
    last_city = 0
    route.append(last_city + 1)  # Add 1 to convert to 1-based indexing

    for _ in range(n - 1):
        for u in range(n):
            if mask & (1 << u) and dp[mask][u] + graph[u][last_city] == dp[mask | (1 << last_city)][last_city]:
                route.append(u + 1)
                mask ^= (1 << last_city)
                last_city = u
                break

    route.append(1)  # Return to the starting city
    return min_distance, route

# ==============================================
# Task 3: Self-Organizing Map (SOM) Approach
# ==============================================

def som_tsp(graph, num_neurons=100, max_iter=1000, learning_rate=0.5):
    n = len(graph)
    # Initialize neurons randomly
    neurons = np.random.rand(num_neurons, 2)
    # Generate random city positions (for simplicity)
    cities = np.random.rand(n, 2)

    for iteration in range(max_iter):
        for city in cities:
            # Find the closest neuron (winner)
            distances = np.linalg.norm(neurons - city, axis=1)
            winner = np.argmin(distances)

            # Update neurons in the neighborhood
            for i in range(num_neurons):
                distance_to_winner = np.linalg.norm(neurons[i] - neurons[winner])
                if distance_to_winner < learning_rate:
                    neurons[i] += learning_rate * (city - neurons[i])

        # Decay learning rate
        learning_rate *= 0.99

    # Extract the route from the neurons
    route = []
    for city in cities:
        distances = np.linalg.norm(neurons - city, axis=1)
        route.append(np.argmin(distances) + 1)  # Add 1 to convert to 1-based indexing

    # Calculate the total distance of the route
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += graph[route[i] - 1][route[i + 1] - 1]
    total_distance += graph[route[-1] - 1][route[0] - 1]  # Return to the starting city

    return total_distance, route

# ==============================================
# Main Execution
# ==============================================

if __name__ == "__main__":
    # Task 2: Classical TSP Solution
    print("Classical TSP Solution (Dynamic Programming):")
    min_distance_dp, route_dp = tsp_dp(graph)
    print(f"Minimum TSP Distance: {min_distance_dp}")
    print(f"Optimal Route: {route_dp}")

    # Task 3: SOM Approach
    print("\nSelf-Organizing Map (SOM) Approach:")
    distance_som, route_som = som_tsp(graph)
    print(f"Approximate TSP Distance: {distance_som}")
    print(f"Approximate Route: {route_som}")