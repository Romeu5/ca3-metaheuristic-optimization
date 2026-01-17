This repository contains the implementation of a Simulated Iterated Local Search (SimILS) algorithm for solving the Traveling Salesman Problem (TSP) with stochastic edge weights. The code includes classes for representing TSP instances, performing local search, and executing the SimILS algorithm. Additionally, it provides functionality for estimating tour costs using Monte Carlo simulations and visualizing the results.

Requirements
- Python 3.x
- NumPy
- Matplotlib

This is part of an assignment of the coursework Metaheuristic Optimizations for the Master's program Computational and Mathematical Engineering at Universitat Oberta de Catalunya (UOC) and Universitat Rovira i Virgili (URV).

It consists on three main files:
- tsp.py: Contains the TSP class with methods for simulating tour lengths.
- simils.py: Implements the SimILS algorithm and local search methods.
- main.py: The main script to run the SimILS algorithm on a TSP instance and visualize results.