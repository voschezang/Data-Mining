import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
import seaborn as sns
import scipy
import scipy.stats
import util.data
import numpy as np
np.random.seed(123)


def savefig(fn, transparent=False,  bbox_inches='tight'):
    # save a 2d plot
    # for latex, use .eps
    # otherwise, add the arg: dpi=300
    # %mkdir - p img
    fn = 'img/' + fn + '.pdf'
    plt.savefig(fn, transparent=transparent, bbox_inches=bbox_inches)


def correlation_grid(data, keys, conditional_x=False, numerical=True):
    rcParams['font.size'] = 14
    include_diagonal = numerical

    n_keys = len(keys)
    if not include_diagonal:
        n_keys -= 1

    with plt.style.context(('ggplot')):
        figsize = (3 * n_keys, 2 * n_keys)
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
        cmap = 'terrain'
        # x
        for i_x, k_x in enumerate(keys):
            # y
            for i_y, k_y in enumerate(keys):
                plot = False
                if include_diagonal and i_x <= i_y:
                    plot = True
                    ax = fig.add_subplot(
                        n_keys, n_keys, i_x + i_y * n_keys + 1)
                elif not include_diagonal and i_x < i_y:
                    plot = True
                    ax = fig.add_subplot(
                        n_keys, n_keys, i_x + (i_y - 1) * n_keys + 1)

                if plot:
                    im = correlation_grid_cell(ax, data, i_x, k_x, i_y, k_y,
                                               n_keys, conditional_x,
                                               numerical, fig, cmap=cmap)
                elif i_x == n_keys - 1 and i_y == 0:
                    # add colorbar
                    assert im is not None, 'iteraration direction is incorrect'
                    width = 2
                    i = i_x + i_y * n_keys + 1
                    print(n_keys, i)
                    ax = plt.subplot2grid(
                        (n_keys, i), (0, n_keys - width), colspan=width)
                    # ax = fig.add_subplot(
                    #     n_keys, n_keys, i_x + i_y * n_keys + 1)
                    colormap(cmap, ax)
                    plt.title('Proportion')
                    # plt.colorbar(im, cax=ax, use_gridspec=True,
                    #              orientation='horizontal', pad=1)

        # cb()
        # ax = grid[-1]
        # cbar = ax.cax.colorbar(im)
        # cbar = grid.cbar_axes[0].colorbar(im)

        # plt.subplots_adjust(bottom=0.1, right=-1, top=0.1)
        # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        # plt.colorbar(im, cax=cax, orientation='horizontal', use_gridspec=True)
        # plt.colorbar(im, use_gridspec=True)

        # colorbar is not compatible with tight layout
        # https://matplotlib.org/users/tight_layout_guide.html

        plt.tight_layout()


def correlation_grid_cell(ax, data, i_x, k_x, i_y, k_y, n_keys,
                          conditional_x, numerical, fig, cmap):
    # include_diagonal = numerical
    if i_x == 0:
        plt.ylabel(k_y)
    if i_y == n_keys:
        plt.xlabel(k_x)

    if not numerical:
        # categories_x = data[k_x].unique()
        # categories_y = data[k_y].unique()
        # summarize
        summary = util.data.summarize_categorical(
            data, k_x, k_y, conditional_x)
        im = plot_summary(ax, summary, cmap)

        # add colorbar below the grid
        # if i_x == n_keys - 1 and i_y == n_keys:
        # [left, bottom, width, height]
        # ax = fig.add_axes([0.2, -0.02, 0.75 * n_keys, 0.1 / n_keys])
        # cb = plt.colorbar(orientation='horizontal', pad=0.5)
        # plt.colorbar(orientation="horizontal",fraction=0.07,anchor=(1.0,0.0))
        return im

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
    y_labels = list(summary[x_labels[0]].keys())
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
    plt.xticks(np.arange(n_x), x_labels, rotation=rotation)
    rotation = 0
    length = sum([len(label) for label in y_labels])
    if length > 10:
        rotation = 45
    else:
        rotation = 0
    plt.yticks(np.arange(n_y), y_labels, rotation=rotation)
#     plt.subplots_adjust(right=0.9)
    util.plot.show_grid(ax, im)
    return img  # :: AxesImage


def colormap(cmap_name: str, ax, ranges=[0, 1]):
    # orientation = 'horizontal'
    cmap = plt.cm.get_cmap(cmap_name)
    ratio = 10
    ax.imshow([cmap(np.arange(cmap.N))], extent=[0, ratio, 0, 1])
    plt.yticks([])
    n_xticks = 5
    plt.xticks(np.linspace(0, ratio, n_xticks),
               np.linspace(ranges[0], ranges[1], n_xticks))
    plt.grid(False)