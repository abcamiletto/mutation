import numpy as np
from numpy.random import normal
from scipy.integrate import solve_ivp

from .model import model, pack, unpack


class System:
    def __init__(self, X0, l, g, B, a, f, lenght, steps):
        # Storing Inputs
        self.X0 = X0
        self.l = l
        self.g = g
        self.B = B
        self.a = a
        self.f = f
        self.lenght = lenght
        self.steps = steps

        # Helper Variables
        self.history = [X0] * steps
        self.timer = np.random.exponential(scale=1 / self.f)

    def solve(self):
        t = 0
        X = self.X0

        for i in range(1, self.steps):
            next_t = self.lenght * i / self.steps
            X = self.step(X, t, next_t)

            self.history[i] = X
            t = next_t

            # TODO : kill variation with I < 1e-3

            parents = self.check_spawning(X)
            for parent in parents:
                X = self.spawn_variant(X, idx=parent)

        history = format_history(self.history)
        t = np.linspace(0, self.lenght, self.steps)

        return history, t

    def step(self, X, t, next_t):
        sol = solve_ivp(model, (t, next_t), X, args=(self.l, self.g, self.a, self.B))
        X = sol.y[:, -1]
        return X

    def spawn_variant(self, X, idx):
        self.l = np.concatenate([self.l, self.l[idx] + normal(size=(1, 1)) / 10]).clip(min=0)
        self.g = np.concatenate([self.g, self.g[idx] + normal(size=(1, 1)) / 10]).clip(min=0)
        self.a = np.concatenate([self.a, self.a[idx] + normal(size=(1, 1)) / 10]).clip(min=0)
        self.f = np.concatenate([self.f, self.f[idx] + normal(size=(1, 1)) / 10]).clip(min=1e-4)
        self.timer = np.concatenate(
            [self.timer, np.random.exponential(scale=1 / self.f[-1], size=(1, 1))]
        )

        size = self.B.shape[0]
        B = np.zeros((size + 1, size + 1))
        B[:size, :size] = self.B
        B[-1, :] = B[-2, :] + normal(size=(size + 1,)) / 10
        B[:, -1] = B[:, -1] + normal(size=(size + 1,)) / 10
        B[-1, -1] = 0
        self.B = B.clip(min=0)

        S, I, R, W = unpack(X)

        S = np.expand_dims(S, 1)
        # TODO : Remove 1e-3 from I
        I = np.expand_dims(np.append(I, 1e-3), 1)
        R = np.expand_dims(np.append(R, 0), 1)
        W = np.expand_dims(np.append(W, 0), 1)

        X = pack([S, I, R, W])
        return X

    def check_spawning(self, X):
        _, I, _, _ = unpack(X)
        self.timer -= I

        parent_variant = []
        for idx, timer in enumerate(self.timer):
            if timer < 0:
                self.timer[idx] = np.random.exponential(scale=1 / self.f[idx])
                parent_variant.append(idx)
        return parent_variant


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
