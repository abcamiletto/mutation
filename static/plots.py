import matplotlib.pyplot as plt

from .button import customButton

plt.style.use("fivethirtyeight")


def main_plot(ax, y, t, size):
    ax.plot(t, y[:, 1 : size + 1])
    ax.legend([*[f"$I_{{{i+1}}}$" for i in range(size)]])
    ax.set_ylabel("Population")
    ax.set_title("All Variants")


def single_variant_plot(ax, y, t, size, idx):
    # Recover default color used by matplotlib
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # Plotting
    ax.plot(t, y[:, idx + size], "--", linewidth=2, c="rosybrown")
    ax.plot(t, y[:, idx + size * 2], "--", linewidth=2, c="slategray")
    ax.plot(t, y[:, idx], c=cycle[(idx - 1) % len(cycle)])
    ax.legend([f"$W_{{{idx}}}$", f"$R_{{{idx}}}$", f"$I_{{{idx}}}$"])


def plot_results(y, t, pokedex):
    size = round((y.shape[-1] - 1) / 3)

    row = size // 3 + 1
    cols = 3

    fig, axes = plt.subplots(row, cols, figsize=(16, row * 3.75))
    axes = axes.flatten()

    main_plot(axes[0], y, t, size)

    buttons = []

    for idx, ax in enumerate(axes[1:], start=1):
        if idx <= size:
            single_variant_plot(ax, y, t, size, idx)

            # Add a button to show properties
            variant = pokedex[idx - 1]
            button = customButton(ax, variant)
            buttons.append([button])

            # Priting axis name if needed
            if idx // row == row - 1:
                ax.set_xlabel("Time")

            if idx % row == 0:
                ax.set_ylabel("Population")

            # Setting title
            variant_name = idx if variant.parent is None else f"{idx} from {variant.parent+1}"
            ax.set_title(f"Variant {variant_name}")

    fig.tight_layout()

    plt.show()
    return fig
