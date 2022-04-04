from dataclasses import fields
from pprint import pprint

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition


class customButton:
    def __init__(self, ax, variant):
        self.variant = variant

        # Building the button
        button_ax = plt.axes([0, 0, 1, 1])
        ip = InsetPosition(ax, [0.07, 1.02, 0.15, 0.1])  # posx, posy, width, height
        button_ax.set_axes_locator(ip)
        self.button = Button(button_ax, "Info")
        self.button.on_clicked(self.process)

    def process(self, event):
        content = []
        row_labels = []
        for field in fields(self.variant):
            row_labels.append(field.name)
            if field.name != "parent":
                content.append([getattr(self.variant, field.name)])
            else:
                parent = getattr(self.variant, field.name)
                content.append([parent + 1 if parent is not None else parent])

        fig, ax = plt.subplots(figsize=(3, 2.25))
        ax.set_axis_off()
        table = ax.table(
            cellText=content,
            rowLabels=row_labels,
            colLabels=["Value"],
            rowColours=["aliceblue"] * 10,
            colColours=["aliceblue"] * 10,
            cellLoc="center",
            loc="upper left",
            colWidths=[1, 1],
        )
        table.scale(1, 2)
        fig.tight_layout()
        plt.show()
        pprint(self.variant)
