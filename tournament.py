import time

from solver.ode import System
from utils.args import process_tournament_args
from utils.generate import generate_random_exp

if __name__ == "__main__":

    for i in range(1, 100):
        l, g, B, a, f, D, X0 = generate_random_exp(dim=i * 10)

        params = {
            "mutation": False,
            "unit_size": 1e-3,
        }

        # Defining the settings for the simulation
        steps = 100
        lenght = 25

        # Solving the simulation
        system = System(X0, l, g, B, a, f, D, lenght, steps, **params)
        y, t, pokedex = system.solve()
