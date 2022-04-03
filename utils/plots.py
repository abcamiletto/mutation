import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")


def plot_results(y, t, parents):
    size = round((y.shape[-1] - 1) / 3)

    row = size // 3 + 1
    cols = 3

    fig, axes = plt.subplots(row, cols, figsize=(16, row * 3.75))
    axes = axes.flatten()

    axes[0].plot(t, y[:, 0], linewidth=7)
    axes[0].plot(t, y[:, 1 : size + 1])
    axes[0].legend(["S", *[f"$I_{{{i+1}}}$" for i in range(size)]])
    axes[0].set_ylabel("Population")
    axes[0].set_title("All Variations")

    # Recover default color used by matplotlib
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, ax in enumerate(axes[1:], start=1):
        if idx <= size:
            ax.plot(t, y[:, 0])
            ax.plot(t, y[:, idx + size], "--", linewidth=2, c="rosybrown")
            ax.plot(t, y[:, idx + size * 2], "--", linewidth=2, c="slategray")
            ax.plot(t, y[:, idx], c=cycle[(idx) % len(cycle)])
            ax.legend(["S", f"$I_{{{idx}}}$", f"$W_{{{idx}}}$", f"$R_{{{idx}}}$"])

            if idx // row == row - 1:
                ax.set_xlabel("Time")

            if idx % row == 0:
                ax.set_ylabel("Population")

            parent = parents[idx - 1] + 1
            variant_name = idx if parent == 0 else f"{idx} from {parent}"
            ax.set_title(f"Variant {variant_name}")

    fig.tight_layout()

    plt.show()
