
import matplotlib
matplotlib.use('Agg')

import os
import cv2
import random
import dominate
import colorsys
import numpy as np
from argparse import Namespace
from collections import OrderedDict
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from ..basic.operations import chk_d, fet_d
from dominate.tags import meta, h3, table, tr, td, p, a, img, br


########################################################################################################################
# Utils
########################################################################################################################

def random_colors(n, bright=True, shuffle=False):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then convert to RGB.
    """
    # Generator colors
    brightness = 1.0 if bright else 0.7
    hsv = [(i * 1.0 / n, 1, brightness) for i in range(n)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # Shuffle
    if shuffle:
        random.shuffle(colors)
    return colors


def gradient_colors(n, change='red2green'):
    """
    Gradually changes.
    :param n:
    :param change:
    :return:
    """
    ratio = 1.0 - (np.arange(0, n, dtype=np.float32) / n)[:, np.newaxis]
    # 1. red -> green
    if change == 'red2green':
        return np.concatenate([ratio, (1.0 - ratio), np.full_like(ratio, 0)], axis=1)
    # 2. red -> blue
    elif change == 'red2blue':
        return np.concatenate([ratio, np.full_like(ratio, 0), (1.0 - ratio)], axis=1)
    # 3. green -> red
    elif change == 'green2red':
        return np.concatenate([(1.0 - ratio), ratio, np.full_like(ratio, 0)], axis=1)
    # 4. green -> blue
    elif change == 'green2blue':
        return np.concatenate([np.full_like(ratio, 0), ratio, (1.0 - ratio)], axis=1)
    # 5. blue -> red
    elif change == 'blue2red':
        return np.concatenate([(1.0 - ratio), np.full_like(ratio, 0), ratio], axis=1)
    # 6. blue -> green
    elif change == 'blue2green':
        return np.concatenate([np.full_like(ratio, 0), (1.0 - ratio), ratio], axis=1)
    else:
        raise NotImplementedError


def gray2heat(image_numpy, norm_method='minmax', **kwargs):
    """
    :param image_numpy: Should be an image with shape of (H, W, C) or a vector.
    :param norm_method:
    :param kwargs:
    :return:
    """

    def _func_norm(_x, _given_min, _given_max):
        _min, _max = _x.min(), _x.max()
        assert _max != _min and _given_max != _given_min
        return ((_x - _min) / (_max - _min)) * (_given_max - _given_min) + _given_min

    # Normalize
    if norm_method == 'minmax':
        dst = _func_norm(image_numpy, 0, 255)
    elif norm_method == 'bound':
        mini, maxi = kwargs['norm_bound']
        assert maxi > mini
        length = maxi - mini
        # Calculate min
        image_min, image_max = image_numpy.min(), image_numpy.max()
        assert image_min >= mini and image_max <= maxi
        norm_min = (image_min - mini) / length * 255.0
        norm_max = (image_max - mini) / length * 255.0
        # Normalize
        dst = _func_norm(image_numpy, norm_min, norm_max)
    else:
        raise NotImplementedError
    # Convert
    heat_map = cv2.applyColorMap(dst.astype(np.uint8), cv2.COLORMAP_JET)
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB)
    return heat_map


class HTML(object):
    """
        This HTML class allows us to save images and write texts into a single HTML file. It consists of functions such as
        - <add_header> (add a text header to the HTML file),
        - <add_images> (add a row of images to the HTML file),
        - <save> (save the HTML to the disk).
    It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
    """
    def __init__(self, web_dir, title='Visualization', width=256, refresh=0):
        """
        Initialize the HTML classes
        Parameters:
            web_dir (str) -- a directory that stores the webpage.
                HTML file will be created at <web_dir>/index.html, images will be saved at <web_dir>/images/.
            title (str)   -- the webpage name
            width (int) -- images width
            refresh (int) -- how often the website refresh itself; if 0, no refreshing
        """
        # Config.
        self._web_dir, self._img_dir = web_dir, os.path.join(web_dir, 'images')
        self._title, self._width = title, width
        # 1. Mkdirs
        if not os.path.exists(self._img_dir): os.makedirs(self._img_dir)
        # 2. Set HTML.
        self._doc = dominate.document(title=title)
        if refresh > 0:
            with self._doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    @property
    def image_dir(self):
        return self._img_dir

    def add_header(self, text):
        """
        Insert a header to the HTML file
        Parameters:
            text (str) -- the header text
        """
        with self._doc:
            h3(text)

    def add_images(self, ims, txts, links):
        """
        Add images to the HTML file
        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) -- a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        # Create table & add
        new_table = table(border=1, style="table-layout: fixed;")
        self._doc.add(new_table)
        # Set table
        with new_table:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="width:%dpx" % self._width, src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self):
        """
        save the current content to the HMTL file
        """
        with open(os.path.join(self._web_dir, 'index.html'), 'wt') as f:
            f.write(self._doc.render())


########################################################################################################################
# API
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# Visualizing packages
# ----------------------------------------------------------------------------------------------------------------------

class IterVisualizer(object):
    """
    Visualize that saves visuals to directories & display them on a HTML.
        1. Items
        (1) When using HTML to display images, the 'vis_dir' contains the followins items:
            - 'images/' dir: which contains all visualized images.
            - 'index.html': The webpage to show all images.
        (2) When not using HTML to display images, the 'vis_dir' directly contains all visualized images.

        2. File formats
        By calling 'self.save_images(visuals, iter_count)', where 'visuals' is a dict of {'name1': image1, ... }, then
    images will be samed using the name 'ITER[STEP]_NAME', where
        - 'ITER' is the 'iter_prefix' config.
        - 'STEP' is the 'iter_count' arg.
        - 'NAME' is the dict key.
    """
    def __init__(self, vis_dir, show_html_width=-1, iter_prefix='iter'):
        # Configs
        self._iter_prefix = iter_prefix
        # 1. Using HTML to display.
        if show_html_width > 0:
            self._webpage = HTML(vis_dir, width=show_html_width)
            self._iter_container, self._visual_labels = [], []
        # 2. Only saving images.
        else:
            if not os.path.exists(vis_dir): os.makedirs(vis_dir)
            self._webpage = Namespace(image_dir=vis_dir)

    def _get_image_name(self, iter_count, name):
        return '%s[%d]_%s' % (self._iter_prefix, iter_count, name)

    def save_images(self, visuals, iter_count=0, **kwargs):
        # 1. Save to disk
        for name, image in visuals.items():
            save_image(image, os.path.join(self._webpage.image_dir, '%s.png' % self._get_image_name(iter_count, name)))
        # 2. Update for website
        if isinstance(self._webpage, HTML):
            # Update key
            for key in list(visuals.keys()):
                if key not in self._visual_labels:
                    self._visual_labels.append(key)
            # Update iter
            self._iter_container.append(iter_count)
            # Flush website
            if chk_d(kwargs, 'flush_website'):
                self.save_website()

    def save_website(self):
        if not isinstance(self._webpage, HTML): return
        # 1. Save website
        # (1) Add each row
        for n in self._iter_container:
            # (1) Generate display
            ims, txts, links = [], [], []
            for label in self._visual_labels:
                img_path = '%s.png' % self._get_image_name(n, label)
                ims.append(img_path)
                txts.append(label)
                links.append(img_path)
            # (2) Add
            self._webpage.add_header('%s[%.3d]' % (self._iter_prefix, n))
            self._webpage.add_images(ims, txts, links)
        # (2) Save to disk
        self._webpage.save()
        # 2. Reset container
        self._iter_container = []


# ----------------------------------------------------------------------------------------------------------------------
# Unsorted
# ----------------------------------------------------------------------------------------------------------------------

def plot_two_scalars_in_one(scalars1, scalars2, x, xlabel, title, save_path):
    """
    :param scalars1: {data: ..., ylabel: ..., color: ...)
    :param scalars2: {data: ..., ylabel: ..., color: ...)
    :param x:
    :param xlabel:
    :param save_path:
    :param title:
    :return:
    """
    # 1. Init fig
    _, ax1 = plt.subplots(dpi=200)
    # 2. Plotting two figures
    if x is None: x = np.arange(len(scalars1['data']))
    ax1.set_xlabel(xlabel)
    # (1) Part 1
    ax1.set_ylabel(scalars1['ylabel'], color=scalars1['color'])
    ax1.plot(x, scalars1['data'], color=scalars1['color'])
    # (2) Part 2
    ax2 = ax1.twinx()
    ax2.set_ylabel(scalars2['ylabel'], color=scalars2['color'])
    ax2.plot(x, scalars2['data'], color=scalars2['color'])
    # Other setting
    if title is not None: plt.title(title)
    plt.tight_layout()
    # 3. Save
    plt.savefig(save_path)
    plt.close()


def plot_multi_scalars_vert(scalars, save_path, **kwargs):
    """
    :param scalars: List of scalars.
        (x), data, (title), (xlabel), ylabel, (ylim)
    :param save_path:
    :return:
    """
    def _plot_method(_data, **_kwargs):
        if isinstance(_data, tuple) or isinstance(_data, list):
            _data, _min, _max = _data
            _x = _kwargs['x'] if 'x' in _kwargs.keys() else np.arange(len(_data))
            plt.plot(_x, _data, **fet_d(_kwargs, 'label', 'color'))
            plt.fill_between(_x, y1=_min, y2=_min, alpha=0.1)
        else:
            _x = _kwargs['x'] if 'x' in _kwargs.keys() else np.arange(len(_data))
            plt.plot(_x, _data, **fet_d(_kwargs, 'label', 'color'))

    # 1. Init figure
    fig = plt.figure(dpi=200)
    if 'title' in kwargs.keys(): fig.suptitle(kwargs['title'])
    # 2. Multi figures
    for plt_index, scalar in enumerate(scalars):
        plt.subplot(int("%d1%d" % (len(scalars), plt_index + 1)))
        if 'title' in scalar.keys(): plt.title(scalar['title'])
        # Labels
        if 'xlabel' in scalar.keys(): plt.xlabel(scalar['xlabel'])
        plt.ylabel(scalar['ylabel'])
        # Line
        if isinstance(scalar['data'], dict):
            colors = gradient_colors(n=len(scalar['data']))
            for index, (label, data) in enumerate(scalar['data'].items()):
                _plot_method(data, label=label, color=colors[index], **fet_d(scalar, 'x'))
            if not ('legend' in scalar.keys() and not scalar['legend']): plt.legend()
        else:
            _plot_method(scalar['data'], **fet_d(scalar, 'x'))
        # Others
        if 'ylim' in scalar.keys(): plt.ylim(*scalar['ylim'])
    # Show
    plt.tight_layout()
    if 'title' in kwargs.keys():
        if 'top' in kwargs.keys(): plt.subplots_adjust(top=kwargs['top'])
        else: plt.subplots_adjust(top=0.9)
    # Save
    plt.savefig(save_path)
    plt.close()


def plot_elapsed_scalars(scalars, x, xlabel, ylabel, title, save_path):
    """
    Plotting multiple scalars.
    :param scalars: {label: data}
    :param x:
    :param xlabel:
    :param ylabel:
    :param title:
    :param save_path:
    :return:
    """
    # 1. Init figure & setup
    plt.figure(dpi=200)
    if title is not None: plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # 2. Plot multi scalars
    if x is None: x = np.arange(len(list(scalars.values())[0]))
    colors = gradient_colors(len(scalars))
    for index, (k, line) in enumerate(scalars.items()):
        assert len(x) == len(line)
        kwargs = {'label': k} if index in [0, len(scalars) - 1] else {}
        plt.plot(x, line, color=colors[index], **kwargs)
    # 3. Save
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_histogram_scalars(scalars, bins, xlabel, save_path):
    stat = {'avg': scalars.mean(), 'std': scalars.std(ddof=1), 'min': scalars.min(), 'max': scalars.max()}
    # 1. Init figure
    plt.figure(dpi=200)
    # 2. Plot
    # (1) Content
    plt.hist(scalars, bins=bins, weights=np.ones_like(scalars)/len(scalars))
    plt.grid(True, linestyle='--', alpha=0.5)
    # (2) Labels
    plt.xlabel(xlabel)
    plt.ylabel('proportion')
    # (3) Titles
    plt.title(', '.join([
        'range: (%.2f, %.2f)' % (stat['min'], stat['max']), 
        'avg: %.2f' % stat['avg'], 'std: %.2f' % stat['std']]))
    # 3. Save & close
    plt.savefig(save_path)
    plt.close()
    # Return
    return stat


# Using t-SNE to visualize high dimensional representations
def visualize_latent_by_tsne(data, label, title, save_path):
    """
    :param data: Numpy.array. (batch, ...)
    :param label: Numpy.array. (batch, )
    :param title:
    :param save_path:
    :return:
    """
    # Randomly generate colors.
    n_classes = len(set(label))
    colors = random_colors(n_classes)
    # 1. Reduce dimensions to (batch, 2)
    data_tsne = TSNE().fit_transform(data)
    # 2. Visualization.
    # (1) Clustering to n_classes sets.
    # 1> Init result.
    data_dict = {}
    # 2> Traverse each data & label
    for x_tsne, y in zip(data_tsne, label):
        if y not in data_dict.keys(): data_dict.update({y: []})
        data_dict[y].append(x_tsne[np.newaxis])
    # (2) Visualize each class.
    for y, data_class in data_dict.items():
        data_class = np.concatenate(data_class, axis=0)
        plt.scatter(data_class[:, 0], data_class[:, 1], c=colors[y])
    # (3) Title & save
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


# Plotting multi axes with confidence interval
def plot_multi_axes(figsize, multi_scalars, axes_kwargs, save_path, **kwargs):
    """
    :param figsize:
    :param multi_scalars: dict:
        { legend_scalars1: {
            'data': [(x_axes1, y_axes1), (x_axes2, y_axes2), ...],
            'y_bounds': [value_axes1, value_axes2, ...], where 'value' could be N/A or (upper, lower),
            'color': shared color across multi axes of current scalars,
            'linestyle': shared line style across multi axes of current scalars,
            'markersize': linestyle size,
            'irregular': N/A or 'vertical_line', 'horizontal_line', 'scatter' },
          legend_scalars2: {...},
          ... }
    :param axes_kwargs: List of dict. Each dict could have keys: rect, xlabel, ylabel, xlim, ylim.
    :param save_path:
    :param kwargs: Shared kwargs. Could have keys:
        legend: legend_kwargs_dict, e.g., { loc='lower center', ncol=2 }
    :return:
    """
    # 1. Get figure & axes
    fig = plt.figure(figsize=figsize, dpi=500)
    # 2. Plot each axes
    # (1) Init legend
    legend_lines = OrderedDict()
    # (2) Plot
    for axis_index, axis_kwargs in enumerate(axes_kwargs):
        # 1. Get ax
        assert 'rect' in axis_kwargs.keys()
        ax = plt.axes(axis_kwargs['rect'])
        # 2. Plot each line
        for legend, scalars in multi_scalars.items():
            # Check data
            scalars_data_axis = scalars['data'][axis_index]
            if scalars_data_axis is None: continue
            ############################################################################################################
            # Kwargs for plotting current scalar within currect axis
            ############################################################################################################
            plot_kwargs = {}
            if 'markersize' in scalars.keys(): plot_kwargs['markersize'] = scalars['markersize']
            if 'color' in scalars.keys(): plot_kwargs['color'] = scalars['color']
            linestyle = scalars['linestyle'] if 'linestyle' in scalars.keys() else '-'
            ############################################################################################################
            # Plotting
            ############################################################################################################
            # 1. regular
            if 'irregular' not in scalars.keys():
                ########################################################################################################
                # 1. Curve
                ########################################################################################################
                if isinstance(scalars_data_axis, tuple) and isinstance(scalars_data_axis[0], list):
                    # Plot line
                    line, = ax.plot(*scalars_data_axis, linestyle, label=legend, **plot_kwargs)
                    # Plot confidence interval
                    if 'y_bounds' in scalars.keys():
                        upper, lower = scalars['y_bounds'][axis_index]
                        plt.fill_between(scalars_data_axis[0], upper, lower, color=plot_kwargs['color'], alpha=0.1)
                    # Legend
                    legend_lines[legend] = line
                ########################################################################################################
                # 2. Horizontal line
                ########################################################################################################
                elif isinstance(scalars_data_axis, float):
                    # Plot
                    line = ax.axhline(scalars_data_axis, linestyle='--', label=legend, **plot_kwargs)
                    # Legend
                    if legend not in legend_lines.keys(): legend_lines[legend] = line
                else:
                    raise NotImplementedError
            # 2. irregular
            else:
                if scalars['irregular'] == 'vertical_line':
                    # (1) Plot line
                    ax.axvline(scalars_data_axis, linestyle='--', **plot_kwargs)
                    # (2) Show legend
                    if 'text_loc' in scalars.keys():
                        ax.text(
                            *scalars['text_loc'][axis_index], legend,
                            **({} if 'text_kwargs' not in scalars.keys() else scalars['text_kwargs'][axis_index]))
                else:
                    # (1) Plot line
                    ax.axhline(scalars_data_axis, linestyle='--', **plot_kwargs)
                    # (2) Show legend
                    if 'text_loc' in scalars.keys():
                        ax.text(
                            *scalars['text_loc'][axis_index], legend,
                            **({} if 'text_kwargs' not in scalars.keys() else scalars['text_kwargs'][axis_index]))
        # 3. Kwargs for plotting currect axis
        if 'xlabel' in axis_kwargs.keys(): ax.set_xlabel(axis_kwargs['xlabel'])
        if 'ylabel' in axis_kwargs.keys(): ax.set_ylabel(axis_kwargs['ylabel'])
        if 'xlim' in axis_kwargs.keys(): plt.xlim(*axis_kwargs['xlim'])
        if 'ylim' in axis_kwargs.keys(): plt.ylim(*axis_kwargs['ylim'])
        if 'title' in axis_kwargs.keys():
            if isinstance(axis_kwargs['title'], str):
                title, title_kwargs = axis_kwargs['title'], {}
            else:
                title, title_kwargs = axis_kwargs['title']
            ax.set_title(title, **title_kwargs)
    # (3) Set legend
    fig.legend(handles=list(legend_lines.values()), labels=list(legend_lines.keys()),
               **(kwargs['legend'] if 'legend' in kwargs.keys() else {}))
    # 3. Save
    plt.savefig(save_path)
    plt.close()
