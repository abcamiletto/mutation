from solver.ode import System
from static.plots import plot_results
from utils.args import process_args
from utils.storing import save_experiment

# Selecting the starting point and parameters
(l, g, B, a, f, X0), params = process_args()

# Defining the settings for the simulation
steps = 100
lenght = 25
# Solving the simulation
system = System(X0, l, g, B, a, f, lenght, steps, **params)
y, t, pokedex = system.solve()

# Plotting results
fig = plot_results(y, t, pokedex)

if input("Do you want to save the settings of the experiments? [y/N] ") == "y":
    save_experiment(l, g, B, a, X0)
