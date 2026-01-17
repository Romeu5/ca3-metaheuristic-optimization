
import numpy as np 
import random 
import matplotlib.pyplot as plt

class GammaDistribution:

    def __init__(self, mean: float, std_dev: float):
        self.shape = (mean / std_dev) ** 2
        self.scale = (std_dev ** 2) / mean

    def sample(self, generator: np.random.Generator):
        return generator.gamma(self.shape, self.scale)

class TSPInstance:

    def __init__(self, filename, seed = 1):
        self.coords = self._parse_tsp(filename)
        self.mean_distances = self._compute_distance_matrix(self.coords)
        self.distance_distributions = self._compute_gamma_distributions_matrix()
        self.generator = np.random.default_rng(seed)
        self.size = len(self.coords)

    # Parse TSP
    def _parse_tsp(self, filename):
        coords = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            node_section = False
            for line in lines:
                line = line.strip()
                if line == "NODE_COORD_SECTION":
                    node_section = True
                    continue
                if line == "EOF":
                    break
                if node_section:
                    parts = line.split()
                    if len(parts) >= 3:
                        x, y = float(parts [1]), float(parts[2]) 
                        coords.append((x, y))
        return np.array(coords)
    
    # Compute distance matrix
    def _compute_distance_matrix(self, coords):
        n = len(coords)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_matrix[i, j] = np.hypot(coords[i][0] - coords[j][0], 
                                                    coords[i][1] - coords[j][1])
        return dist_matrix

    # Tour length (non-stochastic)
    def deterministic_tour_length(self, tour):
        n = len(tour)
        length = sum(self.mean_distances[tour[i], tour[(i + 1) % n]] for i in range(n))
        return length

    # Compute gamma distributions matrix
    def _compute_gamma_distributions_matrix(self):
        n = len(self.coords)
        gamma_matrix = [[None for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                mean = self.mean_distances[i, j]
                if i*j % 3 == 0:
                    std_dev = 0.3 * mean  # 30% of the mean distance
                elif i*j % 3 == 1:
                    std_dev = 0.1 * mean  # 10% of the mean distance
                else:
                    std_dev = 0.5 * mean  # 50% of the mean distance
                gamma_matrix[i][j] = GammaDistribution(mean, std_dev)
                gamma_matrix[j][i] = gamma_matrix[i][j]  # Symmetric
        return gamma_matrix
    
    # Simulate one tour length
    def simulate_one_tour_length(self, tour):
        n = len(tour)
        if n != self.size:
            raise ValueError(f"Tour length does not match number of cities in TSP instance. Length of tour: {n}, number of cities: {self.size}")
        length = 0.0
        for i in range(n):
            from_city = tour[i]
            to_city = tour[(i + 1) % n]
            distance_generator = self.distance_distributions[from_city][to_city]
            sampled_distance = distance_generator.sample(self.generator)
            length += sampled_distance
        return length

    # Simulate multiple tour lengths
    def simulate_tour(self, tour, num_simulations):
        tours = np.zeros(num_simulations, dtype=np.float32)
        for sim in range(num_simulations):
            tours[sim] = self.simulate_one_tour_length(tour)
        return tours

#
# Deterministic greedy TSP
#
def greedy_tsp(distances, start=0):
    n = len(distances)
    tour = [start]
    unvisited = set (range(n)) - {start}
    while unvisited:
        last = tour[-1]
        next_city = min(unvisited, key = lambda city: distances [last, city]) 
        tour.append(next_city) 
        unvisited.remove(next_city)
    return tour
     