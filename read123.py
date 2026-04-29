# Copyright (c) 2015, Warren Weckesser.  All rights reserved.
# This software is licensed according to the "BSD 2-clause" license.

import numpy as _np
from .core import grid_count as _grid_count
import matplotlib.pyplot as _plt
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from ._common import _common_doc

__all__ = ['eyediagram', 'eyediagram_lines']


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)


def eyediagram_lines(y, window_size, offset=0, **plotkwargs):
    """
    Plot an eye diagram using matplotlib by repeatedly calling the `plot`
    function.
    <common>

    """
    start = offset
    while start < len(y):
        end = start + window_size
        if end > len(y):
            end = len(y)
        yy = y[start:end + 1]
        _plt.plot(_np.arange(len(yy)), yy, 'k', **plotkwargs)
        start = end


eyediagram_lines.__doc__ = eyediagram_lines.__doc__.replace("<common>",
                                                            _common_doc)


def eyediagram(y, window_size, offset=0, colorbar=True, **imshowkwargs):
    """
    Plot an eye diagram using matplotlib by creating an image and calling
    the `imshow` function.
    <common>
    """
    counts = _grid_count(y, window_size, offset)
    counts = counts.astype(_np.float32)
    counts[counts == 0] = _np.nan
    ymax = y.max()
    ymin = y.min()
    yamp = ymax - ymin
    min_y = ymin - 0.05 * yamp
    max_y = ymax + 0.05 * yamp

    _plt.imshow(counts.T[::-1, :],
                extent=[0, 8, min_y, max_y],
                **imshowkwargs)
    ax = _plt.gca()

    ax.set_facecolor('black')
    ax.figure.patch.set_facecolor('black')
    _plt.grid(color='w')

    # Show colorbar if enabled
    if colorbar:
        cbar = _plt.colorbar()
        cbar.ax.yaxis.set_tick_params(color='white')  # Set tick color to white
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')  # Set tick labels color to white

    # Fixing the aspect ratio
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    aspect = (xlim[1] - xlim[0]) / ((ylim[1] - ylim[0]) * 1.0)
    ax.set_aspect(aspect)

    font_prop = FontProperties(family='Calibri', size=12)
    # 设置坐标轴刻度字体
    ax.tick_params(colors='white')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)

    # 设置白色虚线网格
    ax.grid(True, which='both', color='white', linestyle='--', linewidth=0.75)

    # 设置x，y轴字体样式
    ax.set_xlabel('Times', color='white', fontproperties=font_prop)
    ax.set_ylabel('Amplitude', color='white', fontproperties=font_prop)

    # Hide axis
    # ax.axis('off')
    # Adjust margins to bring the plot closer to the window edges
    _plt.subplots_adjust(left=0.04, right=0.995, top=0.94, bottom=0.1)  # Adjust margins

    save_path = './signal_ana/Eye_photo.jpg'
    _plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)  # 保存图像，DPI 为 300


eyediagram.__doc__ = eyediagram.__doc__.replace("<common>", _common_doc)
