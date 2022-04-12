import pandas as pd
import plotly.express as px
import plotly.io as pio
from pyparsing import col

color_palette = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

pio.templates[pio.templates.default].layout.colorway = color_palette


def main_plot(y, t, size, healthy):
    to_plot = [y[:, 0]] + [y[:, idx] for idx in range(1, size + 1)] + [t]
    legend = ["S", *[f"I{i+1}" for i in range(size)], "time"]
    if not healthy:
        del to_plot[0]
        del legend[0]
    data = {key: value for key, value in zip(legend, to_plot)}
    return pd.DataFrame(data), legend


def single_plot(y, t, size, idx, healthy):
    to_plot = [y[:, 0], y[:, idx + size], y[:, idx + size * 2], y[:, idx], t]
    legend = ["S", f"W{idx}", f"R{idx}", f"I{idx}", "time"]
    if not healthy:
        del to_plot[0]
        del legend[0]
    data = {key: value for key, value in zip(legend, to_plot)}
    return pd.DataFrame(data), legend


def plotly_results(y, t, pokedex, idx=0, healthy=True):
    size = round((y.shape[-1] - 1) / 3)

    if idx == 0:
        main_data, legend = main_plot(y, t, size, healthy)
        main_fig = px.line(main_data, x="time", y=legend)
    else:
        main_data, legend = single_plot(y, t, size, idx, healthy)
        main_fig = px.line(main_data, x="time", y=legend)
        main_fig.update_traces(
            patch={"line": {"dash": "dash"}, "opacity": 0.7},
            line=dict(color=color_palette[2]),
            selector={"legendgroup": f"W{idx}"},
        )
        main_fig.update_traces(
            patch={"line": {"dash": "dash"}, "opacity": 0.7},
            line=dict(color=color_palette[-1]),
            selector={"legendgroup": f"R{idx}"},
        )
        main_fig.update_traces(
            line=dict(color=color_palette[3]),
            selector={"legendgroup": f"I{idx}"},
        )

    # This styles the line
    main_fig.update_traces(line=dict(width=5))

    return main_fig
