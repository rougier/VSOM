import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


def __min_birth_max_death(persistence, band=0.0):
    # Look for minimum birth date and maximum death date for plot optimisation
    max_death = 0
    min_birth = persistence[0][1][0]
    for interval in reversed(persistence):
        if float(interval[1][1]) != float("inf"):
            if float(interval[1][1]) > max_death:
                max_death = float(interval[1][1])
        if float(interval[1][0]) > max_death:
            max_death = float(interval[1][0])
        if float(interval[1][0]) < min_birth:
            min_birth = float(interval[1][0])
    if band > 0.0:
        max_death += band
    return (min_birth, max_death)


def _array_handler(a):
    if isinstance(a[0][1], np.float64) or isinstance(a[0][1], float):
        return [[0, x] for x in a]
    else:
        return a


def plot_persistence_barcode(
    persistence=[],
    alpha=0.85,
    max_intervals=1024,
    max_barcodes=1024,
    inf_delta=0.1,
    legend=True,
    colormap=None,
    axes=None,
    fontsize=14,
):
    persistence = _array_handler(persistence)

    if max_intervals > 0 and max_intervals < len(persistence):
        # Sort by life time, then takes only the max_intervals elements
        persistence = sorted(
            persistence,
            key=lambda life_time: life_time[1][1] - life_time[1][0],
            reverse=True,
        )[:max_intervals]

    if colormap is None:
        # colormap = plt.cm.Set1.colors
        colormap = CB_color_cycle
    if axes is None:
        fig, axes = plt.subplots(1, 1)

    persistence = sorted(persistence, key=lambda birth: birth[1][0])

    (min_birth, max_death) = __min_birth_max_death(persistence)
    ind = 0
    delta = (max_death - min_birth) * inf_delta
    # Replace infinity values with max_death + delta for bar code to be more
    # readable
    infinity = max_death + delta
    axis_start = min_birth - delta
    # Draw horizontal bars in loop
    for interval in reversed(persistence):
        if float(interval[1][1]) != float("inf"):
            # Finite death case
            axes.barh(
                ind,
                (interval[1][1] - interval[1][0]),
                height=0.8,
                left=interval[1][0],
                alpha=alpha,
                color=colormap[interval[0]],
                linewidth=0.5,
            )
        else:
            # Infinite death case for diagram to be nicer
            axes.barh(
                ind,
                (infinity - interval[1][0]),
                height=0.8,
                left=interval[1][0],
                alpha=alpha,
                color=colormap[interval[0]],
                linewidth=0.5,
            )
        ind = ind + 1

    if legend:
        dimensions = list(set(item[0] for item in persistence))
        axes.legend(
            handles=[
                mpatches.Patch(color=colormap[dim], label="H"+str(dim))
                for dim in dimensions
            ],
            loc="upper right",
        )

    axes.set_title("Persistence barcode", fontsize=fontsize)

    # Ends plot on infinity value and starts a little bit before min_birth
    axes.axis([axis_start, infinity, 0, ind])
    return axes


def plot_persistence_diagram(
    persistence=[],
    alpha=0.6,
    band=0.0,
    max_intervals=1024,
    max_plots=1024,
    inf_delta=0.1,
    legend=True,
    colormap=None,
    axes=None,
    fontsize=14,
    greyblock=False
):
    persistence = _array_handler(persistence)

    if max_plots != 1000:
        print("Deprecated parameter. It has been replaced by max_intervals")
        max_intervals = max_plots

    if max_intervals > 0 and max_intervals < len(persistence):
        # Sort by life time, then takes only the max_intervals elements
        persistence = sorted(
            persistence,
            key=lambda life_time: life_time[1][1] - life_time[1][0],
            reverse=True,
        )[:max_intervals]

    if colormap is None:
        # colormap = plt.cm.Set1.colors
        colormap = CB_color_cycle
    if axes is None:
        fig, axes = plt.subplots(1, 1)

    (min_birth, max_death) = __min_birth_max_death(persistence, band)
    delta = (max_death - min_birth) * inf_delta
    # Replace infinity values with max_death + delta for diagram to be more
    # readable
    infinity = max_death + delta
    axis_end = max_death + delta / 2
    axis_start = min_birth - delta

    # bootstrap band
    if band > 0.0:
        x = np.linspace(axis_start, infinity, 1000)
        axes.fill_between(x, x, x + band, alpha=alpha, facecolor="red")
    # lower diag patch
    if greyblock:
        axes.add_patch(mpatches.Polygon([[axis_start, axis_start],
                                         [axis_end, axis_start],
                                         [axis_end, axis_end]],
                                        fill=True,
                                        color='lightgrey'))
    # Draw points in loop
    pts_at_infty = False  # Records presence of pts at infty
    for interval in reversed(persistence):
        if float(interval[1][1]) != float("inf"):
            # Finite death case
            axes.scatter(
                interval[1][0],
                interval[1][1],
                alpha=alpha,
                color=colormap[interval[0]],
            )
        else:
            pts_at_infty = True
            # Infinite death case for diagram to be nicer
            axes.scatter(interval[1][0],
                         infinity,
                         alpha=alpha,
                         color=colormap[interval[0]])
    if pts_at_infty:
        # infinity line and text
        axes.plot([axis_start, axis_end],
                  [axis_start, axis_end],
                  linewidth=1.0,
                  color="k")
        axes.plot([axis_start, axis_end],
                  [infinity, infinity],
                  linewidth=1.0,
                  color="k",
                  alpha=alpha)
        # Infinity label
        yt = axes.get_yticks()
        yt = yt[np.where(yt < axis_end)]  # to avoid ticklabel higher than inf
        yt = np.append(yt, infinity)
        ytl = ["%.3f" % e for e in yt]  # to avoid float precision error
        ytl[-1] = r'$+\infty$'
        axes.set_yticks(yt)
        axes.set_yticklabels(ytl, fontsize=14, weight='bold')

    if legend:
        dimensions = list(set(item[0] for item in persistence))
        axes.legend(
            handles=[
                mpatches.Patch(color=colormap[dim], label="H"+str(dim))
                for dim in dimensions
            ]
        )

    axes.set_xlabel("Birth", fontsize=fontsize, weight='bold')
    axes.set_ylabel("Death", fontsize=fontsize, weight='bold')
    axes.set_title("Persistence diagram", fontsize=fontsize)
    # Ends plot on infinity value and starts a little bit before min_birth
    axes.axis([axis_start, axis_end, axis_start, infinity + delta/2])
    return axes


def read_pdgm(fname):
    with open(fname, "rb") as f:
        dgm = pickle.load(f)
    return dgm


def plot_diagrams(dgm_input, dgm_regular, dgm_random):
    fig = plt.figure(figsize=(16, 11))
    fig.subplots_adjust(wspace=0.3, hspace=0.2)
    ax1 = fig.add_subplot(231)
    plot_persistence_barcode(dgm_input, axes=ax1)
    ax1.set_title("")
    ax1.set_xlabel(r"$\alpha$", fontsize=21, weight='bold')
    ticks = ax1.get_yticks().astype('i')
    ax1.set_yticklabels(ticks, fontsize=14, weight='bold')
    xlim = ax1.get_xlim()
    ax1.set_xticks(np.round(np.linspace(0, xlim[1], 3), 3))
    ticks = ax1.get_xticks()
    ax1.set_xticklabels(ticks, fontsize=14, weight='bold')
    K = 1024 + 60
    M = 0
    ax1.text(M, K, 'A',
             va='top',
             ha='left',
             fontsize=18,
             weight='bold')

    ax2 = fig.add_subplot(232)
    plot_persistence_barcode(dgm_regular, axes=ax2)
    ax2.set_title("")
    ax2.set_xlabel(r"$\alpha$", fontsize=21, weight='bold')
    ax2.set_yticks([])
    xlim = ax2.get_xlim()
    ax2.set_xticks(np.round(np.linspace(0, xlim[1], 3), 3))
    ticks = ax2.get_xticks()
    ax2.set_xticklabels(ticks, fontsize=14, weight='bold')
    ax2.text(M, K, 'B',
             va='top',
             ha='left',
             fontsize=18,
             weight='bold')

    ax3 = fig.add_subplot(233)
    plot_persistence_barcode(dgm_random, axes=ax3)
    ax3.set_title("")
    ax3.set_xlabel(r"$\alpha$", fontsize=21, weight='bold')
    ax3.set_yticks([])
    xlim = ax3.get_xlim()
    ax3.set_xticks(np.round(np.linspace(0, xlim[1], 3), 3))
    ticks = ax3.get_xticks()
    ax3.set_xticklabels(ticks, fontsize=14, weight='bold')
    ax3.text(M, K, 'C',
             va='top',
             ha='left',
             fontsize=18,
             weight='bold')

    ax4 = fig.add_subplot(234)
    plot_persistence_diagram(dgm_input, axes=ax4)
    ax4.set_title("")
    xlim = ax4.get_xlim()
    ax4.set_xticks(np.round(np.linspace(0, xlim[1], 3), 3))
    ticks = ax4.get_xticks()
    ax4.set_xticklabels(ticks, fontsize=14, weight='bold')
    # K = ax4.get_xlim()[1] + 0.012  2D-Torus
    K = ax4.get_xlim()[1] + 0.002801
    ax4.text(M, K, 'D',
             va='top',
             ha='left',
             fontsize=18,
             weight='bold')

    xlim_track, ylim_track = [], []
    ax5 = fig.add_subplot(235)
    plot_persistence_diagram(dgm_regular, axes=ax5)
    xlim = ax5.get_xlim()
    ylim = ax5.get_ylim()
    xlim_track.append(xlim)
    ylim_track.append(ylim)
    ax5.set_xticks(np.round(np.linspace(0, xlim[1], 3), 3))
    ticks = ax5.get_xticks()
    ax5.set_xticklabels(ticks, fontsize=14, weight='bold')
    ax5.set_ylabel("")
    ax5.set_title("")
    # K = ax5.get_xlim()[1] + 0.00075   2D-torus
    # K = ax5.get_xlim()[1] + 0.00099   2D-holes
    K = ax5.get_xlim()[1] + 0.006
    ax5.text(M, K, 'E',
             va='top',
             ha='left',
             fontsize=18,
             weight='bold')

    ax6 = fig.add_subplot(236)
    plot_persistence_diagram(dgm_random, axes=ax6)
    xlim = ax6.get_xlim()
    ylim = ax6.get_ylim()
    xlim_track.append(xlim)
    ylim_track.append(ylim)
    ax6.set_xticks(np.round(np.linspace(0, xlim[1], 3), 3))
    ticks = ax6.get_xticks()
    ax6.set_xticklabels(ticks, fontsize=14, weight='bold')
    ax6.set_title("")
    ax6.set_ylabel("")
    # K = ax6.get_xlim()[1] + 0.0013 2D Holes and Torus
    K = ax6.get_xlim()[1] + 0.005
    ax6.text(M, K, 'F',
             va='top',
             ha='left',
             fontsize=18,
             weight='bold')
