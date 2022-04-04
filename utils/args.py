import argparse
import sys

import yaml

from .storing import generate_random_exp, load_experiment, yaml_file


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
        exp = generate_random_exp(args.randomize_dim)
    else:
        exp = load_experiment(args.experiment)

    return exp, not args.no_mutation
