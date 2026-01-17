from tsp import TSPInstance
from simils import TSPTour, SimILSTSP, SimILSConfig
import numpy as np
from time import time

DATA_FOLDER = "../../../../CA1/Exercicis/data files/tsp-data"

def plot_tours(tsp: TSPInstance, tours: list[TSPTour], titles: list[str], simulations: int = 100, save_path: str = None):
    """
    Plot histogram of the different tours' simulated lengths.
    """

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    color_map = plt.get_cmap("tab10")

    for idx, (tour, title) in enumerate(zip(tours, titles)):
        color = color_map(idx % 10)
        samples = tsp.simulate_tour(tour.tour, simulations)
        plt.hist(samples, bins=50, alpha=0.5, label=f"{title} (VaR95: {tour.cost:.2f})", color=color)
        #  Add a vertical line for the estimated cost
        plt.axvline(tour.cost, color=color, linestyle='dashed', linewidth=1)
        print(f"{title} - Tour {tour.tour} - Estimated VaR95: {tour.cost:.2f}")

    plt.xlabel("Tour Length")
    plt.ylabel("Frequency")
    plt.title("Histogram of Simulated Tour Lengths")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    seed = 1

    tsp = TSPInstance(f"{DATA_FOLDER}/berlin52.tsp", seed=seed)
     
    def estimator_mean(samples: np.ndarray):
        return np.mean(samples)
    
    def estimator_percentile_95(samples: np.ndarray):
        return np.quantile(samples, 0.95, method="higher")
    
    estimator = estimator_percentile_95

    # Deterministic best tour with deterministic_tour_length = 7544.37
    deterministic_best_tour = [16, 20, 41, 6, 1, 29, 22, 19, 49, 28, 15, 45, 43, 33, 34, 35, 38, 39, 36, 37, 47, 23, 4, 14, 5, 3, 24, 11, 27, 26, 25, 46, 12, 13, 51, 10, 50, 32, 42, 9, 8, 7, 40, 18, 44, 31, 48, 0, 21, 30, 17, 2]

    config = SimILSConfig(
        ils_iters=150,
        local_search_iters=150,
        estimator=estimator,
        simulations_per_estimation=200
    )

    solver = SimILSTSP(tsp, config=config, seed=seed)
    print(f"\nStarting SimILS with {config.ils_iters} ILS iterations and {config.local_search_iters} local search iterations per ILS iteration...")

    start_time = time()
    solutions = solver.run(start_tour=deterministic_best_tour)
    end_time = time()
    print(f"SimILS completed in {end_time - start_time:.2f} seconds.")

    deterministic_best_cost = solver.estimate(deterministic_best_tour, long=True)

    num_solutions_to_plot = 1
    list_solutions_to_plot = solutions.candidates[:num_solutions_to_plot]
    tours_to_plot = list_solutions_to_plot + [TSPTour(deterministic_best_tour, deterministic_best_cost)]
    titles = [f"SimILS Solution {i+1}" for i in range(num_solutions_to_plot)] + ["Best Deterministic Tour"]

    estimator_name = "mean" if config.estimator == estimator_mean else "var95"
    plot_file = f"plots/simils_{config.ils_iters}_{config.local_search_iters}_{config.simulations_per_estimation}_{estimator_name}_s{seed}.png"

    plot_tours(
        tsp, 
        tours_to_plot,
        titles, 
        simulations=10000,
        save_path=plot_file
    )

if __name__ == "__main__":
    main()