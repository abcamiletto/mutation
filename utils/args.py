import argparse
import sys

import yaml

from .generate import generate_random_exp
from .storing import load_experiment, yaml_file


def process_args():
    description = "Modeling and Simulation tool for mutations in epidemic scenarios"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "-l",
        "--list_experiments",
        action="store_true",
        help="Display all avaible experiments",
    )

    parser.add_argument(
        "-e",
        "--experiment",
        metavar="",
        type=int,
        default=0,
        help="Index of the experiment to use.",
    )

    parser.add_argument(
        "-r",
        "--randomize_dim",
        metavar="",
        type=int,
        default=0,
        help="Dimension of the randomized run. If given, it overrides the -e entry",
    )

    parser.add_argument(
        "-d",
        "--deterministic",
        action="store_true",
        help="If True use deterministic behaviours",
    )

    parser.add_argument(
        "-n",
        "--no-mutation",
        action="store_true",
        help="If given this flag disables mutation",
    )

    parser.add_argument(
        "-s",
        "--sick_size",
        metavar="",
        type=float,
        default=0.1,
        help="Percentage of the population sick at the beginning of the simulation, by default 0.1. Only used in randomized runs.",
    )

    parser.add_argument(
        "-o",
        "--outbreak_size",
        metavar="",
        type=float,
        default=0.001,
        help="Percentage of the population sick when a new variation spawn, by default 1e-3.",
    )

    args = parser.parse_args()

    if args.list_experiments:
        with open(str(yaml_file), "r") as f:
            exps = yaml.load(f, Loader=yaml.FullLoader) or {}

        print("The following experiments are available:")
        for idx, exp in enumerate(exps):
            print(f"\t {idx} : {exp}")
        sys.exit()

    if args.deterministic:
        import random

        import numpy as np

        random.seed(42)
        np.random.seed(42)

    if args.randomize_dim != 0:
        exp = generate_random_exp(args.randomize_dim, args.sick_size)
    else:
        exp = load_experiment(args.experiment)

    params = {
        "mutation": not args.no_mutation,
        "unit_size": args.outbreak_size,
    }

    return exp, params


def process_tournament_args():
    description = "Genetic Tournament among variants. No mutation allowed."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "-r",
        "--rounds",
        metavar="",
        type=int,
        default=10,
        help="Subsequent rounds of the tournament, by default 10",
    )

    parser.add_argument(
        "-s",
        "--starting_points",
        metavar="",
        type=int,
        default=10,
        help="Number of variants to start from",
    )

    args = parser.parse_args()

    params = {"mutation": False}

    return args.rounds, args.starting_points, params
