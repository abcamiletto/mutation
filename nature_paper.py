import pathlib

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp2d
from tqdm import tqdm

from solver.ode import System
from solver.params_util import Variant
from utils.generate import build_starting_point

sim_lenght = 75
steps = round(sim_lenght * 2)


results = []
i_range = 100
j_range = 75
max_l_ratio = 4
max_I0_ratio = 2_000
init_l = 0.3
init_I0 = 0.1

for i in tqdm(range(i_range + 1), desc="i"):
    for j in tqdm(range(j_range + 1), desc="j", leave=False, colour="red"):
        alpha = j / j_range
        beta = i / i_range
        new_l = init_l * (1 + alpha * (max_l_ratio - 1))
        new_I0 = init_I0 / (1 + (max_I0_ratio - 1) * beta)

        var1 = Variant(init_l, 0.1, 0.1, 0.4, 0.0, I0=init_I0)
        var2 = Variant(new_l, 0.1, 0.1, 0.4, 0.0, I0=new_I0)

        starting_point = build_starting_point([var1, var2])
        l, g, B, a, f, D, X0 = starting_point
        system = System(X0, l, g, B, a, f, D, sim_lenght, steps, mutation=False)
        y, t, pokedex = system.solve()

        infections = y[:, 1:3].max(axis=0)
        results.append([new_l / init_l, init_I0 / new_I0, infections[1] / infections[0]])

store = pathlib.Path(__file__).parent / "stored_results"
joblib.dump(results, store / "nature.pkl")


x_data = np.array([r[0] for r in results])
y_data = np.array([r[1] for r in results])
z_data = np.array([r[2] for r in results])

df = pd.DataFrame(dict(x=x_data, y=y_data, z=z_data))
df["y"] = df["y"].round(1)
df["x"] = df["x"].round(2)

fig, axes = plt.subplots(1, 2, figsize=(15, 8))

table = df.pivot("y", "x", "z")
table = table.iloc[::-1]
heatmap = sns.heatmap(table, ax=axes[0], xticklabels=25, yticklabels=30)
cbar = heatmap.collections[0].colorbar
cbar.ax.set_ylabel(r"$\frac{maxI_2}{maxI_1}$", rotation=0, fontsize=16)
cbar.ax.yaxis.set_label_coords(2.2, 0.5)


axes[0].set_xlabel("$\lambda_2 / \lambda_1$", fontsize=16)
axes[0].set_ylabel(r"$\frac{I_{01}}{I_{02}}$", rotation=0, fontsize=20)

labels = [item.get_text() for item in axes[0].get_yticklabels()]
axes[0].set_yticklabels([str(round(float(label))) for label in labels])
axes[0].tick_params(axis="y", labelrotation=45)

df["z"] = df["z"] > 1
table = df.pivot("y", "x", "z")
table = table.iloc[::-1]

cmap = sns.color_palette("rocket", 5, as_cmap=True)
cmap = [tuple(cmap.get_under()[:3].tolist()), tuple(cmap.get_over()[:3].tolist())]
heatmap = sns.heatmap(
    table,
    cmap=cmap,
    ax=axes[1],
    xticklabels=25,
    yticklabels=30,
)
cbar = heatmap.collections[0].colorbar
cbar.ax.set_ylabel("Winner", rotation=0)

axes[1].set_xlabel("$\lambda_2 / \lambda_1$", fontsize=16)
axes[1].set_ylabel(r"$\frac{I_{01}}{I_{02}}$", rotation=0, fontsize=20)
labels = [item.get_text() for item in axes[1].get_yticklabels()]
axes[1].set_yticklabels([str(round(float(label))) for label in labels])
axes[1].tick_params(axis="y", labelrotation=45)


colorbar = axes[1].collections[0].colorbar
colorbar.set_ticks([0.25, 0.75])
colorbar.set_ticklabels(["$I_1$", "$I_2$"])


plt.show()
