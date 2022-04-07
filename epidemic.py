import time

from solver.ode import System
from static.plots import plot_results
from utils.args import process_args
from utils.storing import save_experiment

# Selecting the starting point and parameters
(l, g, B, a, f, D, X0), params = process_args()

# Defining the settings for the simulation
steps = 100
lenght = 25
# Solving the simulation
tic = time.perf_counter()
system = System(X0, l, g, B, a, f, D, lenght, steps, **params)
y, t, pokedex = system.solve()
toc = time.perf_counter() - tic
print(f"Time needed to simulate the model {toc:.3f}s")

# Plotting results
fig = plot_results(y, t, pokedex)

if input("Do you want to save the settings of the experiments? [y/N] ") == "y":
    save_experiment(l, g, B, a, f, X0)
