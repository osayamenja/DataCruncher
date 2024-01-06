# Data Crunching for Research
import re
import bisect
import math

import numpy as np
from matplotlib import pyplot
import matplotlib.ticker as ticker
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

    axs[1].plot(runtime_x, normalize(runtime_y, 1 / 1000), 'tab:red') # convert to milliseconds
    axs[1].set_ylim(top=3, bottom=0.5)
    axs[1].grid()
    axs[1].set_ylabel(r"Zoomed-in Runtimes $\mathbf{\mathit{(ms)}}$")
    axs[0].set_xticks([])
    axs[1].set_xticks([])
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


def plot_ecdf(data: list, x_title: str, x_unit: str, title:str):
    np_d = np.array(data)
    min_d = np_d.min()

    slowdown_ecdf = ECDF(data)  # seconds to milliseconds
    #  print(np.where(np.array(slowdown_ecdf.x) == 0.5))
    median_cdf = bisect.bisect(slowdown_ecdf.y, 0.5)

    f, ax2 = pyplot.subplots(dpi=300)
    f.suptitle(title, fontweight='bold')

    # ax1.plot(slowdown_ecdf.x, slowdown_ecdf.y)
    # ax1.set_yticks(np.linspace(0, 1, num=11))
    # ax1.set_xlabel(r"Slowdown Factor", fontweight='bold', fontsize=8)
    # ax1.tick_params(axis='x', labelsize=8)
    # ax1.grid()

    ax2.plot(slowdown_ecdf.x, slowdown_ecdf.y, 'tab:green')
    ax2.set_yticks(np.linspace(0, 1, num=11))
    ax2.tick_params(axis='x', labelsize=8)
    ax2.set_xlim(left=min_d, right=8)
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

    stats = (f'$mean$ = {np.array(filtered_data).mean():.2f}{x_unit}\n'
             f'$median$ = {slowdown_ecdf.x[median_cdf]:.2f}{x_unit}')
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    ax2.text(0.8, 0.7, stats, fontsize=8, bbox=bbox,
             transform=ax2.transAxes, horizontalalignment='right')

    ax2.set_xlabel(x_title, fontweight='bold', fontsize=8)
    ax2.grid()
    pyplot.show()


def get_kernel_percentages(data_file: str, delimiter_regex: str, metadata: dict, skip=0) -> tuple[list, list]:
    percentages = []
    for _ in range(len(metadata)):
        percentages.append(0.0)

    summary = open(data_file, "r")
    p = re.compile(delimiter_regex)

    for _ in range(skip):
        summary.readline()

    line = summary.readline()
    non_misc_percentages = 0
    while line:
        split_summary = p.findall(line)
        time_percent = float(split_summary[0])

        if time_percent > 0.0:
            for i in range(10, len(split_summary)):
                stop_search = False
                j = 0
                # This is correct since dict remembers insertion order per https://stackoverflow.com/a/39980744
                for kernel_pattern in metadata.keys():
                    val = split_summary[i]
                    if re.match(kernel_pattern, val):
                        percentages[j] = percentages[j] + time_percent
                        non_misc_percentages = non_misc_percentages + time_percent
                        stop_search = True
                        break
                    j = j + 1
                if stop_search:
                    print(line)
                    break

        line = summary.readline()

    kernel_groups = list(metadata.values())
    kernel_groups.append("Misc")
    percentages.append(100.0 - non_misc_percentages)

    return kernel_groups, percentages


# Source: https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_and_donut_labels.html
def plot_donut_chart(description: list, data: list, title=None):
    fig, ax = pyplot.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"), dpi=300)

    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="->"),
              bbox=bbox_props, zorder=0, va="center")

    # if stats_str is not None and stats_str is str:
    #     stats = stats_str
    #     bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    #     ax.text(1.4, 0.5, stats, fontsize=8, bbox=bbox,
    #             transform=ax.transAxes, horizontalalignment='center')

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate('{}: {:.1f}%'.format(description[i], data[i]), xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)

    ax.set_title(title, fontweight='bold')

    pyplot.show()


def vanilla_plot(data_x: list, data_y: list, ax_title: str):
    fig, ax = pyplot.subplots(dpi=300)
    ax.plot(data_x, data_y, marker="D")
    ax.grid()
    ax.set_title(ax_title, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax.set_xlabel("Num of GPUs", fontdict=dict(fontweight='bold'))
    pyplot.show()


def single_node_four_way_plot(data_x: list, data_y: list, data_y2: list, data_y3: list, data_y4: list, title: str):
    fig, axs = pyplot.subplots(2, 2, figsize=(9, 6), dpi=300)
    fig.suptitle(title, fontweight='bold', fontsize=14)

    axs[0, 0].plot(data_x, data_y, marker="*")
    axs[0, 0].set_xscale('log', base=2)
    axs[0, 0].set_yticks(np.vectorize(lambda t: t * 10)(np.logspace(0, 2, base=2, num=3)))
    axs[0, 0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    axs[0, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%d s"))
    axs[0, 0].set_title("Single Global Batch Processing Time")
    stats = ('Global Size = 256\n'
             'Micro Size = 4')
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    axs[0, 0].text(0.8, 0.8, stats, fontsize=8, bbox=bbox,
                   transform=axs[0, 0].transAxes, horizontalalignment='center')

    axs[1, 0].plot(data_x, data_y2, 'tab:orange', marker="D")
    axs[1, 0].set_title("All-to-All Bytes Transmitted per GPU ")
    axs[1, 0].set_xscale('log', base=2)
    axs[1, 0].set_yscale('log', base=2)
    axs[1, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%d MB"))
    axs[1, 0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

    axs[0, 1].plot(data_x, data_y3, 'tab:green', marker="X")
    axs[0, 1].set_title("GPU Idle Time Percentage")
    axs[0, 1].set_xscale('log', base=2)
    axs[0, 1].set_yticks(np.arange(min(data_y3) - 0.1, max(data_y3) + 0.15, 0.5))
    axs[0, 1].yaxis.set_major_formatter(ticker.PercentFormatter())
    axs[0, 1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

    axs[1, 1].plot(data_x, data_y4, 'tab:red', marker="h")
    axs[1, 1].set_title("All-to-All Average Runtime")
    axs[1, 1].set_xscale('log', base=2)
    axs[1, 1].set_yticks(np.arange(min(data_y4), max(data_y4) + 0.1, 0.1))
    axs[1, 1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    axs[1, 1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f ms"))

    for ax in axs.flat:
        ax.set(xlabel='Num of GPUs')
    fig.tight_layout()
    pyplot.show()


def multi_node_four_way_plot(data_x: list, data_y: list, data_y2: list,
                             data_y3: list, data_y4: list, title: str):
    fig, axs = pyplot.subplots(2, 2, figsize=(9, 6), dpi=300)
    fig.suptitle(title, fontweight='bold', fontsize=12)

    axs[0, 0].plot(data_x, data_y, marker="*")
    axs[0, 0].set_xscale('log', base=2)
    axs[0, 0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    axs[0, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%d s"))
    axs[0, 0].set_title("Single Global Batch Processing Time")
    stats = ('Global Size = 256\n'
             'Micro Size = 4')
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    axs[0, 0].text(0.8, 0.8, stats, fontsize=8, bbox=bbox,
                   transform=axs[0, 0].transAxes, horizontalalignment='center')

    axs[0, 1].plot(data_x, data_y2, 'tab:orange', marker="D")
    axs[0, 1].set_title("All-to-All Bytes Transmitted per GPU ")
    axs[0, 1].set_xscale('log', base=2)
    axs[0, 1].set_yscale('log', base=2)
    axs[0, 1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f MB"))
    axs[0, 1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

    axs[1, 0].plot(data_x, data_y3, 'tab:green', marker="X")
    axs[1, 0].set_title("GPU Idle Time Percentage")
    axs[1, 0].set_xscale('log', base=2)
    axs[1, 0].set_yticks(np.arange(min(data_y3) - 0.4, max(data_y3) + 2.65, 5))
    axs[1, 0].yaxis.set_major_formatter(ticker.PercentFormatter())
    axs[1, 0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

    axs[1, 1].plot(data_x, data_y4, 'tab:red', marker="h")
    axs[1, 1].set_title("All-to-All Average Runtime")
    axs[1, 1].set_xscale('log', base=2)
    axs[1, 1].set_yticks(np.arange(min(data_y4) - 0.4, max(data_y4) + 1, 1))
    axs[1, 1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    axs[1, 1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f ms"))

    for ax in axs.flat:
        ax.set(xlabel='Num of GPUs')
    fig.tight_layout()
    pyplot.show()


def comparison_two_way_plot(x_tick_labels: list, data_y1: list, data_y2: list, title: str):
    fig, axs = pyplot.subplots(1, 2, figsize=(6, 4), dpi=300)
    fig.suptitle(title, fontweight='bold', fontsize=12)
    x = np.linspace(1, len(data_y1), num=len(data_y1))

    axs[0].plot(x, data_y1, 'tab:cyan', marker="d")
    axs[0].set_title("Idle Time")
    axs[0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%dX"))
    axs[0].set_yticks(data_y1)
    axs[0].set_xticks(x, labels=x_tick_labels)

    axs[1].plot(x, data_y2, 'tab:purple', marker="d")
    axs[1].set_title("All-to-All Average Runtime")
    axs[1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%dX"))
    axs[1].set_xticks(x, labels=x_tick_labels)
    axs[1].set_yticks(data_y2)

    for ax in axs.flat:
        ax.set(xlabel='GPUs')

    fig.tight_layout()
    pyplot.show()


def single_line_plot(x_tick_labels: list, aux_x_tick_labels: list, data_y: list, title: str):
    fig, axs = pyplot.subplots(figsize=(6, 5), dpi=300)
    fig.suptitle(title, fontsize=12)
    x = np.linspace(1, len(data_y), num=len(data_y))

    axs.plot(x, data_y, 'tab:green', marker="d")
    axs.set_xticks(x, labels=x_tick_labels)
    axs.set_xlabel("Num of Parameters", fontweight='bold')
    axs.set_ylabel("$A_n$")
    axs.grid()
    stats = "$A_n = 3 \\cdot L$"
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    axs.text(0.3, 0.75, stats, fontsize=12, bbox=bbox,
             transform=axs.transAxes, horizontalalignment='left')

    ax2 = axs.twiny()
    ax2.plot(x, data_y, 'tab:green', marker="d")
    ax2.set_xticks(x, labels=aux_x_tick_labels)
    ax2.set_xlabel(r"Num of Layers $(L)$", fontweight='bold', fontsize=9)
    fig.tight_layout()
    pyplot.show()


def single_node_amdahl_plot(p1: float, p2: float, title: str):
    x = np.logspace(0, 6, base=2, num=7)
    y1 = np.divide(1.0, ((1 - p1) + (p1 / x)))
    y2 = np.divide(1.0, ((1 - p2) + (p2 / x)))

    fig, axs = pyplot.subplots(figsize=(6, 4), dpi=300)
    #axs.plot(x, y2, 'tab:purple', linestyle='dashed', label='Compute, $f=0.932$')
    axs.plot(x, y1, 'tab:cyan', linestyle='dashed', label='All-to-All, $f=0.058$')
    axs.grid()
    axs.set_title(title)
    axs.set_xscale('log', base=2)
    axs.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    #axs.set_yticks(np.linspace(1, 10, num=7))
    axs.set_ylabel("Projected Speedup")
    axs.set_xlabel("Improvement Factor")
    axs.legend()
    pyplot.show()
    pass


def multi_node_amdahl_plot(p1: float, p2: float, title: str):
    x = np.logspace(0, 6, base=2, num=7)
    y1 = np.divide(1.0, ((1 - p1) + (p1 / x)))
    y2 = np.divide(1.0, ((1 - p2) + (p2 / x)))

    fig, axs = pyplot.subplots(figsize=(6, 4), dpi=300)
    axs.plot(x, y1, 'tab:green', linestyle='dashed', label='All-to-All, $f=0.624$')
    axs.plot(x, y2, 'tab:red', linestyle='dashed', label='Compute, $f=0.376$')
    axs.grid()
    axs.set_title(title)
    axs.set_xscale('log', base=2)
    axs.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    axs.set_ylabel("Projected Speedup")
    axs.set_xlabel("Improvement Factor")
    axs.legend()
    pyplot.show()
    pass


if __name__ == '__main__':
    duration_data = get_collective_duration(
        "data/A2A_trace.txt",
        r"(\d+\.?\d+)",
        8,
        2)
    # plot_runtimes(duration_data)
    # plot_ecdf(normalize(duration_data, 1 / 1000))
    plot_ecdf(normalize(duration_data, np.array(duration_data).min()),
              r"Slowdown Factor",
              "X",
              'CDF for All-to-All Slowdown')

    # Commented keywords are included in the misc category.
    # We removed them for a nicer pie chart.
    # https://github.com/pytorch/pytorch/issues/69963
    # includes direct_copy_kernel, HtoD, DtoH, intra/inter D, and memset.
    # k_metadata = {r".+gemm": "GEMM",
    #               r"SendRecv": "All-to-All",
    #               r"dropout": "Dropout",
    #               r"cuApplyLayerNorm|cuComputePartGradGammaBeta": "LayerNorm",
    #               r"softmax": "Softmax",
    #               # r"CudaCodeGen": "CudaCodeGen",
    #               r"mem.+|.*[cC]opy": "Data Transfers",
    #               r"elementwise": "Elementwise",
    #
    #               # r"gatherTopK|radixSortKVInPlace": "MoE Routing",
    #               # r"AllReduce": "AllReduce",
    #               # r"reduce": "Reduction"
    #               }
    #
    # a, b = get_kernel_percentages("report4_kern_sum_cuda_gpu_sum_column.txt",
    #                               r"(\d+\.?\d+|[^\W_]+)",
    #                               k_metadata,
    #                               5)
    # plot_donut_chart(a, b, "GPT-3-MoE 350M Training Step Distribution: 2 GPUs")
    #
    # k_metadata_coalesced = {r"SendRecv": "All-to-All",
    #                         # https://github.com/pytorch/pytorch/issues/69963
    #                         # includes direct_copy_kernel, HtoD, DtoH, intra/inter D, and memset.
    #                         r"mem.+|.*[cC]opy": "Data Transfers",
    #                         r".+gemm|cuApplyLayerNorm|cuComputePartGradGammaBeta"
    #                         r"|dropout|softmax|reduce|AllReduce|CudaCodeGen"
    #                         r"|elementwise|gatherTopK|radixSortKVInPlace": "Compute",
    #                         }
    #
    # a, b = get_kernel_percentages("report4_kern_sum_cuda_gpu_sum_column.txt",
    #                               r"(\d+\.?\d+|[^\W_]+)",
    #                               k_metadata_coalesced,
    #                               5)
    # plot_donut_chart(a, b, "GPT-3-MoE 350M Training Step Summary: 2 GPUs")

    # k_metadata = {
    #     r"softmax": "Softmax",
    #     r"dropout": "Dropout",
    #     r".+gemm": "GEMM",
    #
    #
    #
    #     r"cuApplyLayerNorm|cuComputePartGradGammaBeta": "LayerNorm",
    #     r"SendRecv": "All-to-All",
    #
    #
    #
    #
    #     # r"CudaCodeGen": "CudaCodeGen",
    #     r"mem.+|.*[cC]opy": "Data Transfers",
    #     r"elementwise": "Elementwise",
    #
    #     # r"gatherTopK|radixSortKVInPlace": "MoE Routing",
    #     # r"AllReduce": "AllReduce",
    #     # r"reduce": "Reduction"
    # }
    # a, b = get_kernel_percentages("multi_node_cuda_gpu_sum.txt",
    #                               r"(\d+\.?\d+|[^\W_]+)",
    #                               k_metadata,
    #                               5)
    # plot_donut_chart(a, b, "GPT3-MoE 350M Training Step Distribution: 4x4 GPUs")
    #
    # print(list(zip(a, b)))
    # vanilla_plot([2, 4, 8], [90.4, 92.2, 92.7], "Single-node Compute Kernel Time Percentages")
    # four_way_plot([2, 4, 8],
    #               [39.3, 21, 10],
    #               [8, 4, 2],
    #               [9.4, 7.7, 7.1],
    #               [0.9, 1.068, 1.059],
    #               'Single-Node GPU Scaling for GPT3-MoE 350M')
    # k_metadata_coalesced = {
    #     # https://github.com/pytorch/pytorch/issues/69963
    #     # includes direct_copy_kernel, HtoD, DtoH, intra/inter D, and memset.
    #
    #     r".+gemm|cuApplyLayerNorm|cuComputePartGradGammaBeta"
    #     r"|dropout|softmax|reduce|AllReduce|CudaCodeGen"
    #     r"|elementwise|gatherTopK|radixSortKVInPlace": "Compute",
    #
    #     r"SendRecv": "All-to-All",
    #     # r"mem.+|.*[cC]opy": "Data Transfers",
    #
    # }
    #
    # a, b = get_kernel_percentages("multi_node_cuda_gpu_sum.txt",
    #                               r"(\d+\.?\d+|[^\W_]+)",
    #                               k_metadata_coalesced,
    #                               5)
    # plot_donut_chart(a, b, "GPT-3-MoE 350M Training Step Summary: 4x4 GPUs")
    # multi_node_four_way_plot([8, 16, 32],
    #                          [8.834, 5, 3.5],
    #                          [2, 1, 0.5],
    #                          [50.4, 61.3, 67.4],
    #                          [7.4, 12.7, 14.7],
    #                          'Multi-Node GPU Scaling for GPT3-MoE 350M')
    # comparison_two_way_plot(["8", "2x4"],
    #                         [1, 7],
    #                         [1, 7],
    #                         "Normalized Comparison of Single- and Multi-Node")
    # single_line_plot(["125M", "350M", "2.7B", "13B", "175B"],
    #                  ["12", "24", "32", "40", "96"],
    #                  [36, 72, 96, 120, 288],
    #                  "All-to-All Kernels ($A_n$) per GPT-3 MoE Training Step")
    #multi_node_amdahl_plot(0.624, 0.376, "Projected Speedup for Multi-Node Optimization")
    #single_node_amdahl_plot(0.058, 0.932, "Projected Speedup for Single-Node Optimization")
    pass
