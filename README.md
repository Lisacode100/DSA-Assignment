# DSA-Assignment
# Task 1: TSP Representation and Data Structures 
# 1.1 Graph Representation
# The graph provided in Figure 1 consists of 7 cities (nodes) with distances labeled on the edges. 
# We need to choose an appropriate data structure to represent this graph.

# Choice of Data Structure
#  A 2D array where the cell at the intersection of row i and column j represents the distance between city i and city j. 
# This is a good choice for dense graphs where most cities are connected.

# Adjacency List:
# A list of lists where each city has a list of its neighboring cities and the corresponding distances.
# This is more space-efficient for sparse graphs.

# Given that the graph is small (7 cities), an adjacency matrix is a suitable choice
# because it allows for efficient distance lookups in constant time 
O
(
1
)
O(1).


