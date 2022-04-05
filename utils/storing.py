import pathlib

import numpy as np
import yaml

yaml_file = pathlib.Path(__file__).parent.parent.joinpath("experiments.yaml")
yaml_file.touch(exist_ok=True)


def save_experiment(l, g, B, a, X0):
    """Saving the current experiment into the experiments.yaml file"""

    name = input("Input a name : ")

    # Reading it
    with open(str(yaml_file), "r") as f:
        exps = yaml.load(f, Loader=yaml.FullLoader) or {}

    size = round((X0.shape[0] - 1) / 3)
    S0 = X0[0]
    I0 = X0[1 : size + 1]
    R0 = X0[size + 1 : size * 2 + 1]
    W0 = X0[size * 2 + 1 : size * 3 + 1]

    params = {
        "lambda": l.tolist(),
        "gamma": g.tolist(),
        "beta": B.tolist(),
        "alpha": a.tolist(),
        "S0": [S0.item()],
        "I0": I0.tolist(),
        "R0": R0.tolist(),
        "W0": W0.tolist(),
    }
    exps[name] = params
    with open(str(yaml_file), "w") as f:
        yaml.dump(exps, f, default_flow_style=False)


def load_experiment(exp=None, file=None):
    """Load experiment of index exp from yaml file"""

    # If the index of the experiment is given
    if exp is not None:
        with open(str(yaml_file), "r") as f:
            exps = yaml.load(f, Loader=yaml.FullLoader) or {}

        name = [key for key in exps][exp]

        params = exps[name]
    # If the parameters sets is given
    elif file is not None:
        params = file

    l, g, B, a, f = (
        params["lambda"],
        params["gamma"],
        params["beta"],
        params["alpha"],
        params["frequency"],
    )

    l = np.expand_dims(np.array(l), 1)
    g = np.expand_dims(np.array(g), 1)
    a = np.expand_dims(np.array(a), 1)
    f = np.expand_dims(np.array(f), 1).clip(min=1e-6)
    B = np.array(B)

    X0 = np.array([*params["S0"], *params["I0"], *params["R0"], *params["W0"]])

    return l, g, B, a, f, X0
