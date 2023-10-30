# Data Crunching for Research
import re
import bisect
import math

import numpy as np
from matplotlib import pyplot
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib.transforms import (Bbox, TransformedBbox,
                                   blended_transform_factory)
from mpl_toolkits.axes_grid1.inset_locator import (BboxConnector,
                                                   BboxConnectorPatch,
                                                   BboxPatch)


def get_collective_duration(data_file: str, delimiter_regex: str, num_gpus: int, index: int, has_header=True) -> list:
    durations = open(data_file, "r")
    result = []
    p = re.compile(delimiter_regex)
    if has_header:
        durations.readline()  # skip header

    line = durations.readline()

    while line:
        max_duration = 0.0
        for i in range(num_gpus):  # Obtain longest duration from concurrent executions
            duration = (float(p.findall(line)[index - 1]) / 1000000.0)  # microseconds to seconds
            max_duration = max(duration, max_duration)
            line = durations.readline()

        result.append(max_duration)
    return result


def normalize(data: list, norm: float) -> list:
    for i in range(len(data)):
        data[i] = data[i] / norm
    return data


def plot_runtimes(data: list):
    np_duration = np.array(data)
    min_duration = np_duration.min()
    max_duration = np_duration.max()
    mean_duration = np_duration.mean()

    runtime_x = np.linspace(1, len(data), num=len(data))
    runtime_y = data

    fig, axs = pyplot.subplots(2, dpi=300)
    fig.suptitle('Runtimes of All-to-All in Training GPT-3 350M MoE on NDv2', fontweight='bold')

    axs[0].plot(runtime_x, runtime_y)
    axs[0].grid()
    axs[0].set_ylabel(r"Runtimes $\mathbf{\mathit{(s)}}$")
    stats = (f'$\\max$ = {max_duration:.2f} s\n'
             f'$\\min$ = {(min_duration * 1000):.2f}$\\>ms$\n'
             f'$\\mu$ = {(mean_duration * 1000):.2f}$\\>ms$')
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    axs[0].text(0.45, 0.7, stats, fontsize=8, bbox=bbox,
                transform=axs[0].transAxes, horizontalalignment='right')

    axs[1].plot(runtime_x, normalize(runtime_y, 1 / 1000), 'tab:red')
    axs[1].set_ylim(top=3, bottom=0.5)
    axs[1].grid()
    axs[1].set_ylabel(r"Runtimes $\mathbf{\mathit{(ms)}}$")
    pyplot.show()


# Source: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/axes_zoom_effect.html
def connect_bbox(bbox1, bbox2,
                 loc1a, loc2a, loc1b, loc2b,
                 prop_lines, prop_patches=None):
    if prop_patches is None:
        prop_patches = {
            **prop_lines,
            "alpha": prop_lines.get("alpha", 1) * 0.2,
            "clip_on": False,
        }

    c1 = BboxConnector(
        bbox1, bbox2, loc1=loc1a, loc2=loc2a, clip_on=False, **prop_lines)
    c2 = BboxConnector(
        bbox1, bbox2, loc1=loc1b, loc2=loc2b, clip_on=False, **prop_lines)

    bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches)

    p = BboxConnectorPatch(bbox1, bbox2,
                           loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                           clip_on=False,
                           **prop_patches)

    return c1, c2, bbox_patch1, bbox_patch2, p


# Source: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/axes_zoom_effect.html
def zoom_effect01(axis1, axis2, xmin, xmax, **kwargs):
    """
    Connect *ax1* and *ax2*. The *xmin*-to-*xmax* range in both axes will
    be marked.

    Parameters
    ----------
    axis1
        The main axes.
    axis2
        The zoomed axes.
    xmin, xmax
        The limits of the colored area in both plot axes.
    **kwargs
        Arguments passed to the patch constructor.
    """

    b_box = Bbox.from_extents(xmin, 0, xmax, 1)

    mybbox1 = TransformedBbox(b_box, axis1.get_xaxis_transform())
    mybbox2 = TransformedBbox(b_box, axis2.get_xaxis_transform())

    prop_patches = {**kwargs, "ec": "none", "alpha": 0.2}

    c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox(
        mybbox1, mybbox2,
        loc1a=3, loc2a=2, loc1b=4, loc2b=1,
        prop_lines=kwargs, prop_patches=prop_patches)

    axis1.add_patch(bbox_patch1)
    axis2.add_patch(bbox_patch2)
    axis2.add_patch(c1)
    axis2.add_patch(c2)
    axis2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p


def plot_ecdf(data: list):
    np_d = np.array(data)
    min_d = np_d.min()

    slowdown_ecdf = ECDF(normalize(data, min_d))
    #  print(np.where(np.array(slowdown_ecdf.x) == 0.5))
    median_cdf = bisect.bisect(slowdown_ecdf.y, 0.5)

    f, (ax1, ax2) = pyplot.subplots(1, 2, dpi=300)
    f.suptitle('CDF for All-to-All Slowdown Factor', fontweight='bold')

    ax1.plot(slowdown_ecdf.x, slowdown_ecdf.y)
    ax1.set_yticks(np.linspace(0, 1, num=11))
    ax1.set_xlabel(r"Slowdown Factor", fontweight='bold', fontsize=8)
    ax1.tick_params(axis='x', labelsize=8)
    ax1.grid()

    ax2.plot(slowdown_ecdf.x, slowdown_ecdf.y, 'tab:green')
    ax2.set_yticks(np.linspace(0, 1, num=11))
    ax2.tick_params(axis='x', labelsize=8)
    ax2.set_xlim(left=1, right=5)
    #  ax2.axvline(x=slowdown_ecdf.x[median_cdf], linewidth=1, ymax=0.5, color='#ff2400')
    # float("{:.1f}".format(slowdown_ecdf.x[median_cdf]))
    ax2.annotate('Median',
                 xy=(float("{:.1f}".format(slowdown_ecdf.x[median_cdf])),
                     float("{:.2f}".format(slowdown_ecdf.y[median_cdf]))),
                 xycoords='data',
                 xytext=(0.3, .38), textcoords='axes fraction',
                 va='top', ha='left',
                 arrowprops=dict(facecolor='red', width=3, shrink=0.05))

    filtered_data = [v for v in slowdown_ecdf.x if not math.isnan(v) and not math.isinf(v)]

    stats = (f'$mean$ = {np.array(filtered_data).mean():.2f}\n'
             f'$median$ = {slowdown_ecdf.x[median_cdf]:.2f}')
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    ax2.text(0.8, 0.7, stats, fontsize=8, bbox=bbox,
             transform=ax2.transAxes, horizontalalignment='right')

    ax2.set_xlabel(r"Slowdown Factor (Zoomed-in)", fontweight='bold', fontsize=8)
    ax2.grid()
    pyplot.show()


if __name__ == '__main__':
    duration_data = get_collective_duration(
        "A2A_trace.txt",
        r"(\d+\.?\d+)",
        8,
        2)

    #  Uncomment either of the following to generate the plots.
    # plot_runtimes(duration_data)
    # plot_ecdf(duration_data)
