import numpy as np
from scipy.integrate import solve_ivp

from .model import model, pack, unpack


def solve(X0, l, g, B, a, lenght, steps, variants=True):
    # Solve ode
    history = [X0] * steps
    current_t = 0
    X = X0

    for i in range(1, steps):
        new_t = lenght * i / steps
        sol = solve_ivp(model, (current_t, new_t), X, args=(l, g, a, B))
        X = sol.y[:, -1]
        history[i] = X
        current_t = new_t

        if variants and (i % round(steps / 5) == 0):
            X, l, g, B, a = spawn_variant(X, l, g, B, a, variant_idx=0)

    history = format_history(history)
    t = np.linspace(0, lenght, steps)

    return history, t


def spawn_variant(X, l, g, B, a, variant_idx):
    l_new = np.expand_dims(l[variant_idx] + np.random.normal() / 10, 1)
    g_new = np.expand_dims(g[variant_idx] + np.random.normal() / 10, 1)
    a_new = np.expand_dims(a[variant_idx] + np.random.normal() / 10, 1)
    l = np.concatenate([l, l_new])
    g = np.concatenate([g, g_new])
    a = np.concatenate([a, a_new])

    size = B.shape[0]
    new_B = np.zeros((size + 1, size + 1))
    new_B[:size, :size] = B
    B = new_B
    B[-1, :] = B[-2, :] + np.random.normal(size=(size + 1,)) / 10
    B[:, -1] = B[:, -1] + np.random.normal(size=(size + 1,)) / 10
    B[-1, -1] = 0
    B = B.clip(min=0)

    S, I, R, W = unpack(X)

    S = np.expand_dims(S, 1)
    I = np.expand_dims(np.append(I, 1e-3), 1)
    R = np.expand_dims(np.append(R, 0), 1)
    W = np.expand_dims(np.append(W, 0), 1)

    X = pack([S, I, R, W])

    return X, l, g, B, a


def format_history(history):
    max_shape = max(history, key=lambda x: x.shape[0]).shape[0]
    max_size = round((max_shape - 1) / 3)

    for idx in range(len(history)):
        X = history[idx]
        current_size = round((X.shape[0] - 1) / 3)
        if current_size < max_size:
            diff = max_size - current_size

            S, I, R, W = unpack(X)
            I = np.append(I, [None] * diff)
            R = np.append(R, [None] * diff)
            W = np.append(W, [None] * diff)
            X = pack([S, I, R, W])

            history[idx] = X

    history = np.squeeze(np.array(history))
    return history


class Solver:
    def __init__(self, X0, l, g, B, a, lenght, steps):
        # Storing Inputs
        self.X0 = X0
        self.l = l
        self.g = g
        self.B = B
        self.a = a
        self.lenght = lenght
        self.steps = steps

        # Helper Variables
        self.history = [X0] * steps

    def solve(self):
        t = 0
        X = self.X0

        for i in range(1, self.steps):
            next_t = self.lenght * i / self.steps
            X = self.step(t, next_t)

            history[i] = X
            t = next_t

            if self.check_spawning():
                X, l, g, B, a = spawn_variant(X, l, g, B, a, variant_idx=0)

        history = format_history(self.history)
        t = np.linspace(0, self.lenght, self.steps)

        return history, t

    def step(self, t, next_t):
        sol = solve_ivp(model, (t, next_t), X, args=(self.l, self.g, self.a, self.B))
        X = sol.y[:, -1]

    def check_spawning(self):
        pass
