import numpy as np
from numpy.random import normal
from scipy.integrate import solve_ivp

from solver.state import append_state, delete_states, get_abs_idx

from .model import model, pack, unpack
from .util import Variant, augment_parameters, create_register, delete_parameters


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
        self.l, self.g, self.B, self.a, self.f, self.timer = augment_parameters(
            self.l, self.g, self.B, self.a, self.f, self.timer, idx
        )

        # Getting the real idx of the parent and updating the pokedex
        real_idx = get_abs_idx(self.history[step], X[idx + 1])

        self.pokedex.append(
            Variant(
                round(self.l[-1].item(), 5),
                round(self.g[-1].item(), 5),
                round(self.B[-1, -1].item(), 5),
                round(self.a[-1].item(), 5),
                round(self.f[-1].item(), 5),
                round(real_idx - 1, 5),
            )
        )

        # Extending the state
        X = append_state(X, idx, self.UNIT)
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
                real_idx = get_abs_idx(self.history[step], infected, self.UNIT)

                # We had cases of numerical issue on np.where, but so rare it's not worth digging it up more
                self.extinguished.append(real_idx - 1)
                self.extinguished.sort()
                # Appending the relative index
                to_delete.append(idx)

        return to_delete

    def delete_state(self, X, idxes):
        """Delete variants from state and parameters"""
        self.l, self.g, self.B, self.a, self.f, self.timer = delete_parameters(
            self.l, self.g, self.B, self.a, self.f, self.timer, idxes
        )

        X = delete_states(X, idxes)
        return X

    def add_to_history(self, step, X):
        """Add state to history, keeping track of absolute indexes"""
        total_size = len(self.extinguished) * 3 + len(X)
        n_variants = round((total_size - 1) / 3)
        complete_state = [0] * total_size

        # Setting extinguished variants indexes to None
        for idx in self.extinguished:
            complete_state[idx + 1] = np.nan
            complete_state[n_variants + idx + 1] = np.nan
            complete_state[2 * n_variants + idx + 1] = np.nan

        # Filling the yet unused elements with the current state
        i = 0
        for j in range(len(complete_state)):
            if not np.isnan(complete_state[j]):
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
            I = np.append(I, [np.nan] * diff)
            R = np.append(R, [np.nan] * diff)
            W = np.append(W, [np.nan] * diff)
            X = pack([S, I, R, W])

            history[idx] = X

    history = np.squeeze(np.array(history))
    return history
