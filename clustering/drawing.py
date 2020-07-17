import numpy as np
import matplotlib as plt
import matplotlib.cm as cm
import scipy.cluster.hierarchy as sch


def configure_colour_map():
    cmap = cm.tab10(np.linspace(0, 1, 10))
    sch.set_link_color_palette([plt.colors.rgb2hex(rgb[:3]) for rgb in cmap])