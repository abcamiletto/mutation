import numpy as np
from numpy.random import normal
from scipy.integrate import solve_ivp

from .model import model, pack, unpack


class System:
    UNIT = 1e-3

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
        self.extinguished = []

    def solve(self):
        t = 0
        X = self.X0

        for i in range(1, self.steps):
            # Calculating next step
            next_t = self.lenght * i / self.steps
            X = self.step(X, t, next_t)

            # self.history[i] = X
            self.add_to_history(i, X)
            t = next_t

            # Spawning new variation
            parents = self.check_spawning(X)
            for parent in parents:
                print("Spawning")
                X = self.spawn_variant(X, idx=parent)

            # Deleting bad variation
            extinct = self.check_deletion(X, step=i)
            if extinct:
                print(f"Deleting {len(extinct)}")
                X = self.delete_state(X, extinct)

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
        I[idx] -= self.UNIT
        I = np.expand_dims(np.append(I, self.UNIT), 1)
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

    def check_deletion(self, X, step):
        _, I, _, _ = unpack(X)

        to_delete = []
        for idx, infected in enumerate(I):
            if infected < 0.8 * self.UNIT:
                # Finding the real index of the variant as
                # other ones could already have been deleted
                real_idx = np.where(self.history[step] == infected)
                self.extinguished.append(real_idx[0].item() - 1)
                self.extinguished.sort()
                to_delete.append(idx)
        return to_delete

    def delete_state(self, X, idxes):
        """Delete variants from state and parameters"""
        self.l = np.delete(self.l, idxes, axis=0)
        self.g = np.delete(self.g, idxes, axis=0)
        self.a = np.delete(self.a, idxes, axis=0)
        self.f = np.delete(self.f, idxes, axis=0)
        self.timer = np.delete(self.timer, idxes, axis=0)
        self.B = np.delete(self.B, idxes, axis=0)
        self.B = np.delete(self.B, idxes, axis=1)

        size = round((X.shape[0] - 1) / 3)
        state_idx = []
        for idx in idxes:
            state_idx.extend([idx + 1, size + idx + 1, 2 * size + idx + 1])

        X = np.delete(X, state_idx)
        return X

    def add_to_history(self, step, X):
        total_size = len(self.extinguished) * 3 + len(X)
        n_variants = round((total_size - 1) / 3)
        complete_state = [0] * total_size

        for idx in self.extinguished:
            complete_state[idx + 1] = None
            complete_state[n_variants + idx + 1] = None
            complete_state[2 * n_variants + idx + 1] = None

        i = 0
        for j in range(len(complete_state)):
            if complete_state[j] is not None:
                complete_state[j] = X[i].item()
                i += 1

        self.history[step] = np.array(complete_state)


def format_history(history):
    """Rewriting history ad posterium to add None at the first steps of mid-born variant"""
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
