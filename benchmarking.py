import time

from solver.ode import System
from utils.generate import generate_random_exp

if __name__ == "__main__":
    time_spent = []

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
        tic = time.perf_counter()
        system = System(X0, l, g, B, a, f, D, lenght, steps, **params)
        y, t, pokedex = system.solve()
        toc = time.perf_counter() - tic
        time_spent.append(toc)
        print(f"Time spent for {i*10} variants: {toc:.3f}")
        # time.sleep(toc * 3)

    import joblib
    import matplotlib.pyplot as plt

    joblib.dump(time_spent, "benchmark_variants.pkl")
    print(f"Saved {len(time_spent)} results")
    plt.plot(time_spent)
    plt.show()
