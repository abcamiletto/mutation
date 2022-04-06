import pandas as pd
import plotly.express as px


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
            selector={"legendgroup": f"W{idx}"},
        )
        main_fig.update_traces(
            patch={"line": {"dash": "dash"}, "opacity": 0.7},
            selector={"legendgroup": f"R{idx}"},
        )

    # This styles the line
    main_fig.update_traces(line=dict(width=5))

    return main_fig
