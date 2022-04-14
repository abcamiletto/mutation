from copy import deepcopy
from pprint import pprint
from random import shuffle

import numpy as np
from tqdm import tqdm

from solver.ode import System
from utils.args import process_tournament_args
from utils.generate import build_starting_point, generate_from_prior, generate_random_vars


def royal_rumble(variants, fitness="letal"):
    starting_point = build_starting_point(variants=variants, sick_size=0.1, use_beta=False)
    sim_lenght = 25
    steps = sim_lenght * 2
    l, g, B, a, f, D, X0 = starting_point
    system = System(X0, l, g, B, a, f, D, sim_lenght, steps, mutation=False)
    y, t, pokedex = system.solve()

    # Calculating the most letal one
    if fitness == "letal":
        infections = y[:, 1 : 1 + len(variants)].sum(axis=0)
        death_rates = np.array([var.dI for var in variants])
        deaths = infections * death_rates
        idx_winner = np.argmax(deaths)
    elif fitness == "contagious":
        infections = y[:, 1 : 1 + len(variants)].max(axis=0)
        idx_winner = np.argmax(infections)

    return variants[int(idx_winner)]


def grouper(list_, n):
    return zip(*[iter(list_)] * n)


if __name__ == "__main__":
    rounds, dimension, params = process_tournament_args()
    # We start with N vars
    init_variants = generate_random_vars(dim=dimension)

    for i in tqdm(range(rounds)):
        pool = deepcopy(init_variants)
        # We add 8*N variants to reach a total of 9*N variants
        for var in init_variants:
            pool.extend(generate_from_prior(8, var, clipped=True))

        # We add N new variants to get to 10*N
        pool.extend(generate_random_vars(dim=dimension))
        shuffle(pool)

        winners = []
        for tens in grouper(pool, 10):
            winner = royal_rumble(tens)
            winners.append(winner)

        init_variants = winners

    winner = royal_rumble(winners)
    pprint(winner)
