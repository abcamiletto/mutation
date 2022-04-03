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
        pprint(self.variant)
