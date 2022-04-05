import numpy as np
from numpy.random import normal
from scipy.integrate import solve_ivp

from .model import model, pack, unpack
from .register import Variant, create_register


class System:
    """Wrapper class to store info about the system evolution"""

    def __init__(self, X0, l, g, B, a, f, lenght, steps, mutation, unit_size=1e-3):
        # Storing Inputs
        self.X0 = X0
        self.l = l
        self.g = g
        self.B = B
        self.a = a
        self.f = f
        self.lenght = lenght
        self.steps = steps
        self.mutation = mutation
        self.UNIT = unit_size  # definition of an outbreak

        # Helper Variables
        self.history = [X0] * steps
        self.timer = np.random.exponential(scale=1 / self.f)  # Timers to spawn new variantsS
        self.extinguished = []  # Absolute index of extinguished variants
        self.pokedex = create_register(l, g, B, a, f)

    def solve(self):
        """Solving the ODEs"""
        # Initial Conditions
        t = 0
        X = self.X0

        for i in range(1, self.steps):

            # Calculating next step
            next_t = self.lenght * i / self.steps
            X = self.step(X, t, next_t)
            t = next_t

            # Updating History
            self.add_to_history(i, X)

            # Spawning new variation
            if self.mutation:
                parents = self.check_spawning(X)
                for parent in parents:
                    X = self.spawn_variant(X, idx=parent, step=i)

                # Deleting bad variation
                extinct = self.check_deletion(X, step=i)
                if extinct:
                    X = self.delete_state(X, extinct)

        # Format History ad posterium
        history = format_history(self.history)

        # Times to evaluate and plot the solution
        t = np.linspace(0, self.lenght, self.steps)

        return history, t, self.pokedex

    def step(self, X, t, next_t):
        """Single RK45 step of ODEs"""
        sol = solve_ivp(model, (t, next_t), X, args=(self.l, self.g, self.a, self.B))
        X = sol.y[:, -1]
        return X

    def spawn_variant(self, X, idx, step):
        """Spawn a new variant from parent idx"""
        # Extending model parameters
        self.l = np.concatenate([self.l, self.l[idx] + normal(size=(1, 1)) / 10]).clip(min=0)
        self.g = np.concatenate([self.g, self.g[idx] + normal(size=(1, 1)) / 10]).clip(min=0)
        self.a = np.concatenate([self.a, self.a[idx] + normal(size=(1, 1)) / 10]).clip(min=0)
        self.f = np.concatenate([self.f, self.f[idx] + normal(size=(1, 1)) / 10]).clip(min=1e-6)
        self.timer = np.concatenate(
            [self.timer, np.random.exponential(scale=1 / self.f[-1], size=(1, 1))]
        )

        size = self.B.shape[0]
        B = np.zeros((size + 1, size + 1))
        B[:size, :size] = self.B
        B[-1, :] = B[-2, :] + normal(size=(size + 1,)) / 10
        B[:, -1] = B[:, -2] + normal(size=(size + 1,)) / 10
        B[-1, -1] = 0
        self.B = B.clip(min=0)

        # Getting the real idx of the parent and updating the pokedex
        real_idx = np.where(self.history[step] == X[idx + 1])
        self.pokedex.append(
            Variant(
                round(self.l[-1].item(), 5),
                round(self.g[-1].item(), 5),
                round(self.B[-1, -1].item(), 5),
                round(self.a[-1].item(), 5),
                round(self.f[-1].item(), 5),
                round(real_idx[0].item() - 1, 5),
            )
        )

        # Extending the state
        S, I, R, W = unpack(X)

        S = np.expand_dims(S, 1)
        I[idx] -= self.UNIT
        I = np.expand_dims(np.append(I, self.UNIT), 1)
        R = np.expand_dims(np.append(R, 0), 1)
        W = np.expand_dims(np.append(W, 0), 1)

        X = pack([S, I, R, W])
        return X

    def check_spawning(self, X):
        """Check from whom we should spawn a new variant"""
        _, I, _, _ = unpack(X)
        self.timer -= I

        parent_variant = []
        for idx, timer in enumerate(self.timer):
            if timer < 0:
                # Resetting the timer
                self.timer[idx] = np.random.exponential(scale=1 / self.f[idx])
                parent_variant.append(idx)
        return parent_variant

    def check_deletion(self, X, step):
        """Check which variant we should kill because it's extinguished"""
        _, I, R, W = unpack(X)

        to_delete = []
        for idx, (infected, recovered, weak) in enumerate(zip(I, R, W)):
            if infected + recovered * 0.35 + weak * 0.35 < 0.95 * self.UNIT:
                # Finding the absolute index of the variant
                real_idx = np.where(self.history[step] == infected)

                # It may happen that the variant to delete is one that just spawned
                if real_idx[0].size == 0:
                    # In that case we look for it like this
                    real_idx = np.where(self.history[step] == infected + np.array(self.UNIT))

                # We had cases of numerical issue on np.where, but so rare it's not worth digging it up more
                if real_idx[0].size > 0:
                    self.extinguished.append(real_idx[0].item() - 1)
                    self.extinguished.sort()
                    # Appending the relative index
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
        """Add state to history, keeping track of absolute indexes"""
        total_size = len(self.extinguished) * 3 + len(X)
        n_variants = round((total_size - 1) / 3)
        complete_state = [0] * total_size

        # Setting extinguished variants indexes to None
        for idx in self.extinguished:
            complete_state[idx + 1] = None
            complete_state[n_variants + idx + 1] = None
            complete_state[2 * n_variants + idx + 1] = None

        # Filling the yet unused elements with the current state
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
