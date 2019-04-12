import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import AxesGrid
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
import seaborn as sns
import scipy
import scipy.stats
import itertools
import util.data
import util.string
import numpy as np
np.random.seed(123)


def savefig(fn, transparent=False,  bbox_inches='tight'):
    # save a 2d plot
    # for latex, use .eps
    # otherwise, add the arg: dpi=300
    # %mkdir - p img
    fn = 'img/' + fn + '.pdf'
    plt.savefig(fn, transparent=transparent, bbox_inches=bbox_inches)


def correlation_grid(data, keys, conditional_x=False, numerical=True,
                     cmap='terrain'):
    rcParams['font.size'] = 14
    include_diagonal = numerical

    n_keys = len(keys)
    if not include_diagonal and not conditional_x:
        n_keys -= 1

    with plt.style.context(('ggplot')):
        width = 3 if numerical else 2
        figsize = (width * n_keys, 2 * n_keys)
        figsize = (width * n_keys, 2 * n_keys)
        fig = plt.figure(figsize=figsize)
        # if not numerical:
        #     grid = AxesGrid(fig, 111,
        #                     nrows_ncols=(2, 3),
        #                     axes_pad=0.05,
        #                     cbar_mode='single',
        #                     cbar_location='right',  # bottom
        #                     cbar_pad=0.1
        #                     )
        # else:
        #     fig, (ax, ax_cb) = plt.figure(nrows=2, figsize=figsize,
        #                                   gridspec_kw={"height_ratios":
        #                                                [1, 0.05]})
        # n_keys, 0.1
        # x
        for i_x, k_x in enumerate(keys):
            # y
            for i_y, k_y in enumerate(keys):
                if include_diagonal or conditional_x:
                    assert not (not include_diagonal and not conditional_x)
                    ax = fig.add_subplot(
                        n_keys, n_keys, i_x + i_y * n_keys + 1)
                elif i_x < i_y:
                    assert not include_diagonal and not conditional_x
                    ax = fig.add_subplot(
                        n_keys, n_keys, i_x + (i_y - 1) * n_keys + 1)

                if (include_diagonal and i_x <= i_y) \
                        or (conditional_x and i_x != i_y) \
                        or (not include_diagonal and not conditional_x and
                            i_x < i_y):
                    correlation_grid_cell(ax, data, i_x, k_x,
                                          i_y, k_y, n_keys,
                                          conditional_x,
                                          numerical, fig,
                                          cmap=cmap)

                elif not conditional_x and not numerical and \
                        i_x == n_keys - 1 and i_y == 0:
                    correlation_grid_colorbar(n_keys, i_x, i_y, cmap)
                elif include_diagonal or conditional_x:
                    # hide background
                    hide_ax(ax)

                # add y labels to left-most cells and x labels to bottom cells
                if i_x == 0:
                    plt.ylabel(k_y)
                # if i_y == n_keys - 1:
                if ((numerical or conditional_x) and i_y == n_keys - 1) \
                        or (~numerical and i_y == n_keys):
                    plt.xlabel(k_x)

        fig.align_labels()
        # note that colorbar is not compatible with tight layout
        # https://matplotlib.org/users/tight_layout_guide.html
        plt.tight_layout()


def correlation_grid_colorbar(n_keys, i_x, i_y, cmap):
    # add colorbar
    # assert im is not None, 'iteraration direction is incorrect'
    # plt.colorbar(im, cax=ax, use_gridspec=True,
    #              orientation='horizontal', pad=1)
    # width = 2 if n_keys > 3 else 1
    if n_keys > 4:
        width = 3
    elif n_keys > 3:
        width = 2
    else:
        width = 1
    i = i_x + i_y * n_keys + 1
    ax = plt.subplot2grid(
        (n_keys, i), (0, n_keys - width), colspan=width)
    # ax = fig.add_subplot(
    #     n_keys, n_keys, i_x + i_y * n_keys + 1)
    n_xticks = 9 if n_keys > 4 else 5
    colormap(cmap, ax, ratio=5 * width,  n_xticks=n_xticks)
    plt.title('Proportion')


def correlation_grid_cell(ax, data, i_x, k_x, i_y, k_y, n_keys,
                          conditional_x, numerical, fig, cmap):
    # include_diagonal = numerical

    if not numerical:
        # categories_x = data[k_x].unique()
        # categories_y = data[k_y].unique()
        # summarize
        summary = util.data.summarize_categorical(
            data, k_x, k_y, conditional_x)
        im = plot_summary(ax, summary, cmap=cmap)

        # add colorbar below the grid
        # if i_x == n_keys - 1 and i_y == n_keys:
        # [left, bottom, width, height]
        # ax = fig.add_axes([0.2, -0.02, 0.75 * n_keys, 0.1 / n_keys])
        # cb = plt.colorbar(orientation='horizontal', pad=0.5)
        # plt.colorbar(orientation="horizontal",fraction=0.07,anchor=(1.0,0.0))
        return

    # else:
    x = getattr(data, k_x)
    y = getattr(data, k_y)
    x = util.data.to_floats(x)
    y = util.data.to_floats(y)
    x, y = util.data.filter_nans(x, y)
    if k_x == k_y:
        # plot hist + kde curve (approximated distribution)
        sns.distplot(x, hist=True, kde=True, color='darkgreen')
    else:
        plt.scatter(x, y, s=9, alpha=0.8)
        # compute correlation (symmetric for x,y)
#             x = np.array(x)[~np.isnan(x)]
#             y = np.array(y)[~np.isnan(y)][:x.size]
        r, p = scipy.stats.pearsonr(x, y)
        significant = 'significant' if p < 0.05 else 'not significant'
        print('%s ~ %s: \t %s \t p-value: %0.4f, c: %0.4f' %
              (k_x, k_y, significant, p, r))

        # fit regression line (asymmetric for x,y)
        xy_pred, _result = regression(x, y, line=True, v=0)
        plt.plot(xy_pred[:, 0], xy_pred[:, 1],
                 color='black', lw=1, alpha=0.9)


def show_grid(ax, X):
    # clear grid
    ax.grid(False)
    # disable spines
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    # add custom grid
    # add white grid to distinguish cells
    lw = 3
    ax.set_xticks(np.arange(X.shape[0] + 1) - 0.5 + 0.005 * lw, minor=True)
    ax.set_yticks(np.arange(X.shape[1] + 1) - 0.5 + 0.01 * lw, minor=True)
    ax.grid(which="minor", color='w', linestyle='-', linewidth=lw)  # 3
    ax.tick_params(which="minor", bottom=False, left=False)


def gen_line(x=[0, 1], a=1, b=0):
    # y = ax + b
    x = np.array([np.min(x), np.max(x)])
    return np.array([x, b + a * np.array(x)]).T


def regression(x, y, line=True, v=0):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    result = {}
    result['slope'] = slope
    result['intercept'] = intercept
    result['p_value'] = p_value
    result['std_err'] = std_err
    if v:
        print(result)
    if line:
        xy_pred = gen_line(x, slope, intercept)
    return xy_pred, result


def plot_summary(ax, summary, show_grid=True, cmap='terrain'):
    x_labels = list(summary.keys())
    # not all ylabels may be present in all x-items
    # y_labels = list(summary[x_labels[0]].keys())
    y_labels = [list(summary[x_labels[i]].keys())
                for i in range(len(x_labels))]
    y_labels = list(set(itertools.chain(*y_labels)))

    # TODO fix labels before plotting
    util.data.fix_labels(x_labels)
    util.data.fix_labels(y_labels)
    n_x = len(x_labels)
    n_y = len(y_labels)
    im = np.zeros((n_x, n_y))
    for i, k_x in enumerate(summary.keys()):
        for j, (k_y, v) in enumerate(summary[k_x].items()):
            im[i, j] = v
    # https://matplotlib.org/examples/color/colormaps_reference.html
    # bone plasma rainbow pink cubehelix
    img = plt.imshow(im.T, origin='lower', vmin=0, vmax=1, cmap=cmap)
    rotation = 0
    length = sum([len(label) for label in x_labels])
    if length > 10:
        rotation = 45
    else:
        rotation = 0
#         plt.subplots_adjust(bottom=0.15)
    x_labels = util.string.pad_labels(x_labels)
    plt.xticks(np.arange(n_x), x_labels, rotation=rotation)
    rotation = 0
    length = sum([len(label) for label in y_labels])
    if length > 10:
        rotation = 45
    else:
        rotation = 0
    y_labels = util.string.pad_labels(y_labels)
    plt.yticks(np.arange(n_y), y_labels, rotation=rotation)
#     plt.subplots_adjust(right=0.9)
    util.plot.show_grid(ax, im)
    return img  # :: AxesImage


def categorical_distribution(data: dict, fig=None, xlabel='', sort=True,
                             label_func=lambda x: x):
    # barplot
    if fig is None:
        fig = plt.figure()
    if sort:
        items = sorted(data.items(), key=lambda x: x[0])
    else:
        items = data.items()
    keys, values = zip(*items)
    values = np.array(values) / sum(values)
    keys = list(data.keys())
    keys = [label_func(k) for k in keys]
    keys = util.string.pad_labels(keys)
    plt.bar(range(len(keys)), values, tick_label=keys)
    plt.xlabel(xlabel)
    return fig


def colormap(cmap_name: str, ax, ranges=[0, 1], ratio=9, n_xticks=9):
    """ Simulate a pyplot colorbar
    """
    # orientation = 'horizontal'
    cmap = plt.cm.get_cmap(cmap_name)
    ax.imshow([cmap(np.arange(cmap.N))], extent=[0, ratio, 0, 1])
    plt.yticks([])
    # n_xticks = 5
    plt.xticks(np.linspace(0, ratio, n_xticks),
               np.linspace(ranges[0], ranges[1], n_xticks))

    # minor grid
    plt.grid(False, 'major')
    # n_yticks = 17
    # n_xtick_segments = n_xticks - 1  # + 2
    n_inner_minor_xticks = (n_xticks - 1) * 1
    n_inner_segments = n_inner_minor_xticks  # first and last half counted as one
    segment_width = ratio / n_inner_segments
    start_offset = segment_width / 2
    minor_xticks = np.concatenate([
        [0],
        np.linspace(
            start_offset, ratio - start_offset, n_inner_segments),
        [ratio]])
    ax.set_xticks(minor_xticks, minor=True)
    ax.grid(which="minor", color='w', linestyle='-', linewidth=2.5)
    # ax.set_xticks(minor_xticks, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    # plt.grid(True, 'minor', 'y', ls='--', lw=.5, c='black', alpha=1.)


def hide_ax(ax):
    ax.grid(False)
    ax.patch.set_visible(False)
    plt.xticks([])
    plt.yticks([])
