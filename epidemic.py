from solver.ode import solve
from utils.plots import plot_results
from utils.storing import load_experiment, save_experiment

# Selecting the starting point and parameters
l, g, B, a, X0 = load_experiment()

# Defining the settings for the simulation
steps = 100
lenght = 25
# Solving the simulation
y, t = solve(X0, l, g, B, a, lenght, steps)

# Plotting results
plot_results(y, t)

if input("Do you want to save the settings of the experiments? [y/N] ") == "y":
    save_experiment(l, g, B, a, X0)
