from tsp import TSPInstance
import numpy as np
from dataclasses import dataclass, field

@dataclass
class SimILSConfig:
    ils_iters: int
    local_search_iters: int
    estimator: callable # function of (samples: np.ndarray) -> float
    simulations_per_estimation: int = 20

@dataclass
class TSPTour:
    tour: list = field(default_factory=list)
    cost: float = None

    def copy(self):
        return TSPTour(self.tour.copy(), self.cost)
    
class SolutionCandidateList:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.candidates: list[TSPTour] = []

    def add(self, candidate: TSPTour):
        if len(self.candidates) < self.max_size:
            self.candidates.append(candidate)
        else:
            worst_candidate = self.candidates[-1]
            if candidate.cost < worst_candidate.cost:
                self.candidates[-1] = candidate
        self.candidates.sort(key=lambda x: x.cost)

    def get_best(self) -> TSPTour:
        return self.candidates[0] if self.candidates else None
    
    def __str__(self):
        return str([float(c.cost) for c in self.candidates])

class SimILSTSP:
    def __init__(self, tsp: TSPInstance, config: SimILSConfig, seed = 1):
        self.tsp = tsp
        self.config = config
        self.generator = np.random.default_rng(seed)

    def estimate(self, tour, long = False):
        iterations = self.config.simulations_per_estimation
        if long:
            iterations *= 100
        tour_samples = self.tsp.simulate_tour(tour, num_simulations=iterations)
        return self.config.estimator(tour_samples)
    

    # Implementation of ILS for TSP for a generic permutation and local search 
    def run(self, start_tour = None):
        N = self.tsp.size
        # Construct an init solution. 
        best_sol = TSPTour()
        if start_tour is not None:
            best_sol.tour = start_tour
        else:
            best_sol.tour = self.greedy_randomized_construction()
        best_sol.cost = self.estimate(best_sol.tour, long=True)
        print(f"Initial solution cost: {best_sol.cost}")

        candidate_solutions = SolutionCandidateList(max_size=5)
        candidate_solutions.add(best_sol)

        for iter in range(self.config.ils_iters):
            new_sol = best_sol.copy()
            new_sol = self.local_search(new_sol, self.config.local_search_iters)
            new_sol.cost = self.estimate(new_sol.tour, long=True)

            # Update best solution
            if new_sol.cost < best_sol.cost:
                best_sol = new_sol.copy()
                print(f"Current best found solution cost: {best_sol.cost}, ILS iteration {iter + 1}/{self.config.ils_iters}")

            # Add solution to the list of best solutions
            candidate_solutions.add(new_sol)

        print(best_sol.cost, candidate_solutions)

        local_minima_solutions = SolutionCandidateList(max_size=5)

        for candidate in candidate_solutions.candidates:
            local_min = self.find_local_minimum(candidate)
            local_min.cost = self.estimate(local_min.tour, long=True)
            local_minima_solutions.add(local_min)
            print(f"Went from candidate cost {candidate.cost} found local minimum with cost {local_min.cost}")

        print(local_minima_solutions)

        return local_minima_solutions

    # Local search using long estimates and double of iterations
    def find_local_minimum(self, solution: TSPTour):
        iters = self.config.local_search_iters * 2
        return self.local_search(solution, iters, long=True)
    
    def perturbation(self, solution: TSPTour):
        solution_tour = solution.tour.copy()
        perturbed_solution_tour = self.double_bridge_move(solution_tour)
        cost = self.estimate(perturbed_solution_tour)

        return TSPTour(perturbed_solution_tour, cost)

    def double_bridge_move(self, solution):
        num_slices = 4
        slice_len = len(solution) // num_slices
        p = self.generator.integers(slice_len, size=3) + 1
        p1 = p[0]
        p2 = p1 + p[1]
        p3 = p2 + p[2]
        return solution[:p1] + solution[p3:] + solution[p2:p3] + solution[p1:p2]    

    # Used if no initial solution is provided
    def greedy_randomized_construction(self, alpha=0.5):
        """
        Creates a greedy randomized solution based on the mean value of the distances of the 
        TSP intance
        
        :param tsp: Description
        :type tsp: TSPInstance
        :param alpha: Description
        """
        N = self.tsp.size # Num nodes

        nodes_left = set(range(N))

        # Choose random init node
        start = self.generator.choice(np.arange(N))
        solution = [start]

        nodes_left -= {start}

        while len(nodes_left) > 0:
            # costs to unvisited nodes
            current_node = solution[-1]
            nodes_left_list = list(nodes_left)
            costs_to_unvisited = self.tsp.mean_distances[current_node, nodes_left_list]

            min_cost = np.min(costs_to_unvisited)
            max_cost = np.max(costs_to_unvisited)

            threshold = min_cost + (max_cost - min_cost) * alpha

            rcl = [nodes_left_list[i] for i in range(len(nodes_left_list)) if costs_to_unvisited[i] <= threshold]

            next_node = self.generator.choice(rcl)
            solution.append(next_node)
            nodes_left.remove(next_node)

        return solution

    # Local Search using 2-opt moves
    def local_search(self,  solution: TSPTour, max_iters_local_search, long = False):
        new_solution = solution.copy()
        best_solution = solution.copy()
        count = 0

        while count < max_iters_local_search:
            count += 1
            new_solution = best_solution.copy()
            i, j = self.stochastic_two_opt(new_solution)
            new_solution.tour[i:j+1] = reversed(best_solution.tour[i:j+1])
            new_solution.cost = self.estimate(new_solution.tour, long=long)

            # if we find a lower cost, update the solution
            if new_solution.cost < best_solution.cost:
                # print("found better solution in local search:", new_solution.cost)
                best_solution = new_solution.copy()
                count = 0

        return best_solution

    def stochastic_two_opt(self, solution: TSPTour):
        N = self.tsp.size
        # Select two positions in the tour
        i, j = self.generator.choice(np.arange(N), size=2, replace=False)
        if i > j:
            i, j = j, i

        # Avoid full path reverse, give wrong cost diff
        if i == 0 and j == (N - 1):
            i = 1

        return i, j

