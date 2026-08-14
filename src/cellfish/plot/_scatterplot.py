"""
Scatter plot utilities for single-cell data visualization.

This module provides enhanced scatter plot functionality for single-cell data,
based on the omicverse library (https://github.com/Starlitnightly/omicverse).

Original implementation:
- https://github.com/Starlitnightly/omicverse/blob/main/omicverse/pl/_scatterplot.py

Citation:
    Zhang, X., et al. (2023). omicverse: A comprehensive single-cell analysis toolkit.
    Bioinformatics, 39(1), btac746. https://doi.org/10.1093/bioinformatics/btac746

This implementation has been adapted and enhanced for cellfish.
"""

import collections.abc as cabc
from copy import copy
from numbers import Integral
from itertools import combinations, product
import matplotlib
from typing import (
    Collection,
    Union,
    Optional,
    Sequence,
    Any,
    Mapping,
    List,
    Tuple,
    Literal,
)
from warnings import warn

import numpy as np
import pandas as pd
from anndata import AnnData
from cycler import Cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
# Removed deprecated is_categorical_dtype import
# Using isinstance(dtype, pd.CategoricalDtype) instead
from matplotlib import pyplot as pl, colors, colormaps
from matplotlib import rcParams
from matplotlib import patheffects
from matplotlib.colors import Colormap, Normalize
from functools import partial

from scanpy.plotting import _utils

from scanpy.plotting._utils import (
    _FontWeight,
    _FontSize,
    ColorLike,
    VBound,
    circles,
    check_projection,
    check_colornorm,
)
from scanpy.plotting._docs import (
    doc_adata_color_etc,
    doc_edges_arrows,
    doc_scatter_embedding,
    doc_scatter_spatial,
    doc_show_save_ax,
)
from scanpy import logging as logg
from scanpy._settings import settings
from scanpy._utils import sanitize_anndata, _doc_params, Empty, _empty

def _get_vector_friendly():
    """Get the vector_friendly setting from omicverse plot settings."""
    try:
        from ._plot import _vector_friendly
        return _vector_friendly
    except ImportError:
        try:
            return settings._vector_friendly
        except AttributeError:
            return True  # Default fallback

@_doc_params(
    adata_color_etc=doc_adata_color_etc,
    edges_arrows=doc_edges_arrows,
    scatter_bulk=doc_scatter_embedding,
    show_save_ax=doc_show_save_ax,
)
def embedding(
    adata: AnnData,
    basis: str,
    *,
    color: Union[str, Sequence[str], None] = None,
    gene_symbols: Optional[str] = None,
    use_raw: Optional[bool] = None,
    sort_order: bool = True,
    edges: bool = False,
    edges_width: float = 0.1,
    edges_color: Union[str, Sequence[float], Sequence[str]] = 'grey',
    neighbors_key: Optional[str] = None,
    arrows: bool = False,
    arrows_kwds: Optional[Mapping[str, Any]] = None,
    groups: Optional[str] = None,
    components: Union[str, Sequence[str]] = None,
    dimensions: Optional[Union[Tuple[int, int], Sequence[Tuple[int, int]]]] = None,
    layer: Optional[str] = None,
    projection: Literal['2d', '3d'] = '2d',
    scale_factor: Optional[float] = None,
    color_map: Union[Colormap, str, None] = None,
    cmap: Union[Colormap, str, None] = None,
    palette: Union[str, Sequence[str], Cycler, None] = None,
    na_color: ColorLike = "lightgray",
    na_in_legend: bool = True,
    size: Union[float, Sequence[float], None] = None,
    frameon: Optional[bool] = None,
    legend_fontsize: Union[int, float, _FontSize, None] = None,
    legend_fontweight: Union[int, _FontWeight] = 'bold',
    legend_loc: str = 'right margin',
    legend_numbered: bool = False,
    legend_on_data_numbers: bool = False,
    legend_number_start: int = 1,
    legend_title: Optional[str] = None,
    # Custom right-margin legend renderer. This keeps the scatter/on-data
    # numbered labels unchanged, and only changes the legend panel.
    legend_style: str = "default",
    legend_groupby: Optional[Union[str, Mapping[str, str]]] = None,
    legend_group_order: Optional[Sequence[str]] = None,
    legend_columns: Optional[Sequence[Sequence[str]]] = None,
    legend_marker_map: Optional[Mapping[str, str]] = None,
    legend_edge_palette: Optional[Mapping[str, ColorLike]] = None,
    legend_ncols: Optional[int] = None,
    # Key layout knobs: fonts/sizes/gaps are derived relatively from these.
    legend_scale: float = 1.0,
    legend_density: float = 1.0,
    legend_panel_width: float = 0.92,
    legend_panel_pad: float = 0.035,
    legend_badge_edgewidth: float = 0.0,
    legend_badge_edgecolor: Optional[ColorLike] = None,
    legend_label_sep: str = " | ",
    legend_show_border: bool = False,
    legend_fontoutline: Optional[int] = None,
    colorbar_loc: Optional[str] = "right",
    colorbar_width: Optional[float] = None,
    colorbar_pad: Optional[float] = None,
    colorbar_height_fraction: float = 0.3,
    vmax: Union[VBound, Sequence[VBound], None] = None,
    vmin: Union[VBound, Sequence[VBound], None] = None,
    vcenter: Union[VBound, Sequence[VBound], None] = None,
    norm: Union[Normalize, Sequence[Normalize], None] = None,
    add_outline: Optional[bool] = False,
    outline_width: Tuple[float, float] = (0.3, 0.05),
    outline_color: Tuple[str, str] = ('black', 'white'),
    ncols: int = 4,
    hspace: float = 0.25,
    wspace: Optional[float] = None,
    title: Union[str, Sequence[str], None] = None,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    ax: Optional[Axes] = None,
    return_fig: Optional[bool] = None,
    marker: Union[str, Sequence[str]] = '.',
    arrow_scale: float = 10, 
    arrow_width: float = 0.005,
    **kwargs,
) -> Union[Figure, Axes, None]:
    r"""Scatter plot for user specified embedding basis (e.g. umap, pca, etc).

    Arguments:
        adata: Annotated data matrix
        basis: Name of the obsm basis to use
        color: Keys for annotations of observations/cells or variables/genes (None)
        gene_symbols: Column name in .var DataFrame for gene symbols (None)
        use_raw: Whether to use .raw attribute of adata (None)
        sort_order: Sort order for points by color values (True)
        edges: Whether to draw edges of graph (False)
        edges_width: Width of edges (0.1)
        edges_color: Color of edges ('grey')
        neighbors_key: Key to use for neighbors (None)
        arrows: Whether to draw arrows (False) 
        arrows_kwds: Keywords for arrow plotting (None)
        groups: Restrict to a subset of groups (None)
        components: Components to plot (None)
        dimensions: Dimensions to plot (None)
        layer: Layer to use for coloring (None)
        projection: Projection type - '2d' or '3d' ('2d')
        scale_factor: Scaling factor for spatial coordinates (None)
        color_map: Colormap for continuous variables (None)
        cmap: Alias for color_map (None)
        palette: Colors to use for categorical variables (None)
        na_color: Color for missing values ('lightgray')
        na_in_legend: Include missing values in legend (True)
        size: Point size (None)
        frameon: Draw frame around plot (None)
        legend_fontsize: Font size for legend (None)
        legend_fontweight: Font weight for legend ('bold')
        legend_loc: Location of legend ('right margin')
        legend_fontoutline: Font outline width for legend (None)
        colorbar_loc: Location of colorbar ('right')
        colorbar_width: Colorbar width as a figure fraction when set; auto-scales
            from panel width when omitted.
        colorbar_pad: Gap between panel and colorbar, as a fraction of panel width
            (not figure width). Default is 0.015 × panel width.
        colorbar_height_fraction: Colorbar height as a fraction of panel height.
        vmax: Maximum color scale value (None)
        vmin: Minimum color scale value (None)
        vcenter: Center color scale value (None)
        norm: Normalization for color scale (None)
        add_outline: Add outline to points (False)
        outline_width: Width of outline (0.3, 0.05)
        outline_color: Color of outline ('black', 'white')
        ncols: Number of columns for multi-panel plots (4)
        hspace: Height spacing between subplots (0.25)
        wspace: Width spacing between subplots (None)
        title: Plot title (None)
        show: Show the plot (None)
        save: Save the plot (None)
        ax: Matplotlib axes object (None)
        return_fig: Return figure object (None)
        marker: Marker style ('.') 
        **kwargs: Additional arguments passed to scatter
        
    Returns:
        Matplotlib axes or figure object if show=False
    """
    #####################
    # Argument handling #
    #####################

    check_projection(projection)
    sanitize_anndata(adata)

    basis_values = _get_basis(adata, basis)
    dimensions = _components_to_dimensions(
        components, dimensions, projection=projection, total_dims=basis_values.shape[1]
    )
    args_3d = dict(projection='3d') if projection == '3d' else {}

    # Figure out if we're using raw
    if use_raw is None:
        # check if adata.raw is set
        use_raw = layer is None and adata.raw is not None
    if use_raw and layer is not None:
        raise ValueError(
            "Cannot use both a layer and the raw representation. Was passed:"
            f"use_raw={use_raw}, layer={layer}."
        )
    if use_raw and adata.raw is None:
        raise ValueError(
            "`use_raw` is set to True but AnnData object does not have raw. "
            "Please check."
        )

    if isinstance(groups, str):
        groups = [groups]

    # Color map
    if color_map is not None:
        if cmap is not None:
            raise ValueError("Cannot specify both `color_map` and `cmap`.")
        else:
            cmap = color_map
    if matplotlib.__version__ < "3.7.0":
        if cmap is not None:
            pass
        else: 
            cmap = 'RdBu_r'
        if type(cmap)==matplotlib.colors.LinearSegmentedColormap:
            pass
        else:
            cmap = copy(colormaps.get_cmap(cmap))
            cmap.set_bad(na_color)
    else:
        if cmap is not None:
            pass
        else: 
            cmap = 'RdBu_r'
        if type(cmap)==matplotlib.colors.LinearSegmentedColormap:
            pass
        else:
            cmap = copy(matplotlib.colormaps[cmap])
            cmap.set_bad(na_color)

    
    kwargs["cmap"] = cmap
    # Prevents warnings during legend creation
    na_color = colors.to_hex(na_color, keep_alpha=True)

    if 'edgecolor' not in kwargs:
        # by default turn off edge color. Otherwise, for
        # very small sizes the edge will not reduce its size
        # (https://github.com/scverse/scanpy/issues/293)
        kwargs['edgecolor'] = 'none'

    # Vectorized arguments

    # turn color into a python list
    color = [color] if isinstance(color, str) or color is None else list(color)

    # turn marker into a python list
    marker = [marker] if isinstance(marker, str) else list(marker)

    if title is not None:
        # turn title into a python list if not None
        title = [title] if isinstance(title, str) else list(title)

    # turn vmax and vmin into a sequence
    if isinstance(vmax, str) or not isinstance(vmax, cabc.Sequence):
        vmax = [vmax]
    if isinstance(vmin, str) or not isinstance(vmin, cabc.Sequence):
        vmin = [vmin]
    if isinstance(vcenter, str) or not isinstance(vcenter, cabc.Sequence):
        vcenter = [vcenter]
    if isinstance(norm, Normalize) or not isinstance(norm, cabc.Sequence):
        norm = [norm]

    # Size
    if 's' in kwargs and size is None:
        size = kwargs.pop('s')
    if size is not None:
        # check if size is any type of sequence, and if so
        # set as ndarray
        if (
            size is not None
            and isinstance(size, (cabc.Sequence, pd.Series, np.ndarray))
            and len(size) == adata.shape[0]
        ):
            size = np.array(size, dtype=float)
    else:
        size = 120000 / adata.shape[0]

    ##########
    # Layout #
    ##########
    # Most of the code is for the case when multiple plots are required

    if wspace is None:
        #  try to set a wspace that is not too large or too small given the
        #  current figure size
        wspace = 0.75 / rcParams['figure.figsize'][0] + 0.02

    if components is not None:
        color, dimensions = list(zip(*product(color, dimensions)))

    color, dimensions, marker = _broadcast_args(color, dimensions, marker)

    # 'color' is a list of names that want to be plotted.
    # Eg. ['Gene1', 'louvain', 'Gene2'].
    # component_list is a list of components [[0,1], [1,2]]
    if (
        not isinstance(color, str)
        and isinstance(color, cabc.Sequence)
        and len(color) > 1
    ) or len(dimensions) > 1:
        if ax is not None:
            raise ValueError(
                "Cannot specify `ax` when plotting multiple panels "
                "(each for a given value of 'color')."
            )

        # each plot needs to be its own panel
        fig, grid = _panel_grid(hspace, wspace, ncols, len(color))
    else:
        grid = None
        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(111, **args_3d)

    ############
    # Plotting #
    ############
    axs = []

    # use itertools.product to make a plot for each color and for each component
    # For example if color=[gene1, gene2] and components=['1,2, '2,3'].
    # The plots are: [
    #     color=gene1, components=[1,2], color=gene1, components=[2,3],
    #     color=gene2, components = [1, 2], color=gene2, components=[2,3],
    # ]
    for count, (value_to_plot, dims) in enumerate(zip(color, dimensions)):
        color_source_vector = _get_color_source_vector(
            adata,
            value_to_plot,
            layer=layer,
            use_raw=use_raw,
            gene_symbols=gene_symbols,
            groups=groups,
        )
        color_vector, categorical = _color_vector(
            adata,
            value_to_plot,
            color_source_vector,
            palette=palette,
            na_color=na_color,
        )
        def _is_numeric_array(x):
            arr = np.asarray(x)
            return np.issubdtype(arr.dtype, np.number)

        # Order points
        order = slice(None)
        if sort_order is True and value_to_plot is not None and (categorical is False) and _is_numeric_array(color_vector):
            # Continuous values: higher values on top
            arr = np.asarray(color_vector)
            order = np.argsort(-arr, kind="stable")[::-1]
        elif sort_order and (categorical or not _is_numeric_array(color_vector)):
            # Categorical or non-numeric (strings/colors): missing values sink
            order = np.argsort(~pd.isnull(color_source_vector), kind="stable")
        # Set orders
        if isinstance(size, np.ndarray):
            size = np.array(size)[order]
        color_source_vector = color_source_vector[order]
        color_vector = color_vector[order]
        coords = basis_values[:, dims][order, :]

        # if plotting multiple panels, get the ax from the grid spec
        # else use the ax value (either user given or created previously)
        if grid:
            ax = pl.subplot(grid[count], **args_3d)
            axs.append(ax)
        if frameon ==False:
            ax.axis('off')
            if projection != '3d':
                from ..plot._embedding import add_arrow
                add_arrow(ax,adata,basis,fontsize=legend_fontsize,arrow_scale=arrow_scale,arrow_width=arrow_width)
        elif frameon == 'small':
            ax.axis('off')
            if projection != '3d':
                from ..plot._embedding import add_arrow
                add_arrow(ax,adata,basis,fontsize=legend_fontsize,arrow_scale=arrow_scale,arrow_width=arrow_width)
            '''
            #ax.axis('off')
            xmin=coords[:, 0].min()
            xmax=coords[:, 0].max()
            ymin=coords[:, 1].min()
            ymax=coords[:, 1].max()

            #ax.spines['left'].set_position(('outward', 10))
            #ax.spines['bottom'].set_position(('axes', 0))
            ax.spines['left'].set_position(('data', xmin))
            ax.spines['bottom'].set_position(('data', ymin))

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #ax.spines['bottom'].set_bounds(xmin,xmin+(xmax-xmin)/6)
            #ax.spines['left'].set_bounds(ymin,ymin+(ymax-ymin)/6)
            '''


        
        if title is None:
            if value_to_plot is not None:
                ax.set_title(value_to_plot)
            else:
                ax.set_title('')
        else:
            try:
                ax.set_title(title[count])
            except IndexError:
                logg.warning(
                    "The title list is shorter than the number of panels. "
                    "Using 'color' value instead for some plots."
                )
                ax.set_title(value_to_plot)

        if not categorical:
            vmin_float, vmax_float, vcenter_float, norm_obj = _get_vboundnorm(
                vmin, vmax, vcenter, norm, count, color_vector
            )
            normalize = check_colornorm(
                vmin_float,
                vmax_float,
                vcenter_float,
                norm_obj,
            )
        else:
            normalize = None

        # make the scatter plot
        if projection == '3d':
            cax = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=color_vector,
                rasterized=_get_vector_friendly(),
                norm=normalize,
                marker=marker[count],
                **kwargs,
            )
        else:
            scatter = (
                partial(ax.scatter, s=size, plotnonfinite=True)
                if scale_factor is None
                else partial(
                    circles, s=size, ax=ax, scale_factor=scale_factor
                )  # size in circles is radius
            )

            if add_outline:
                # the default outline is a black edge followed by a
                # thin white edged added around connected clusters.
                # To add an outline
                # three overlapping scatter plots are drawn:
                # First black dots with slightly larger size,
                # then, white dots a bit smaller, but still larger
                # than the final dots. Then the final dots are drawn
                # with some transparency.

                bg_width, gap_width = outline_width
                point = np.sqrt(size)
                gap_size = (point + (point * gap_width) * 2) ** 2
                bg_size = (np.sqrt(gap_size) + (point * bg_width) * 2) ** 2
                # the default black and white colors can be changes using
                # the contour_config parameter
                bg_color, gap_color = outline_color

                # remove edge from kwargs if present
                # because edge needs to be set to None
                kwargs['edgecolor'] = 'none'

                # remove alpha for outline
                alpha = kwargs.pop('alpha') if 'alpha' in kwargs else None

                ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    s=bg_size,
                    c=bg_color,
                    rasterized=_get_vector_friendly(),
                    norm=normalize,
                    marker=marker[count],
                    **kwargs,
                )
                ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    s=gap_size,
                    c=gap_color,
                    rasterized=_get_vector_friendly(),
                    norm=normalize,
                    marker=marker[count],
                    **kwargs,
                )
                # if user did not set alpha, set alpha to 0.7
                kwargs['alpha'] = 0.7 if alpha is None else alpha

            cax = scatter(
                coords[:, 0],
                coords[:, 1],
                c=color_vector,
                rasterized=_get_vector_friendly(),
                norm=normalize,
                marker=marker[count],
                **kwargs,
            )

        # remove y and x ticks
        ax.set_yticks([])
        ax.set_xticks([])
        if projection == '3d':
            ax.set_zticks([])

        # set default axis_labels (e.g. X_umap → "UMAP 1", "UMAP 2")
        name = _basis2name(basis)
        axis_labels = [f"{name} {d + 1}" for d in dims]

        ax.set_xlabel(axis_labels[0],loc='left',fontsize=legend_fontsize)
        ax.set_ylabel(axis_labels[1],loc='bottom',fontsize=legend_fontsize)
        if projection == '3d':
            # shift the label closer to the axis
            ax.set_zlabel(axis_labels[2], labelpad=-7)
        ax.autoscale_view()

        # After plotting scatter

        if edges:
            _utils.plot_edges(ax, adata, basis, edges_width, edges_color, neighbors_key)
        if arrows:
            _utils.plot_arrows(ax, adata, basis, arrows_kwds)

        if value_to_plot is None:
            # if only dots were plotted without an associated value
            # there is not need to plot a legend or a colorbar
            continue

        if legend_fontoutline is not None:
            path_effect = [
                patheffects.withStroke(linewidth=legend_fontoutline, foreground='w')
            ]
        else:
            path_effect = None

        # Adding legends
        if categorical or color_vector.dtype == bool:
            legend_groupby_vector = None
            legend_groupby_mapping = None
            if legend_groupby is not None:
                if isinstance(legend_groupby, str):
                    legend_groupby_vector = _get_color_source_vector(
                        adata,
                        legend_groupby,
                        layer=layer,
                        use_raw=False,
                        gene_symbols=None,
                        groups=None,
                    )
                    try:
                        legend_groupby_vector = legend_groupby_vector[order]
                    except Exception:
                        legend_groupby_vector = np.asarray(legend_groupby_vector)[order]
                elif isinstance(legend_groupby, cabc.Mapping):
                    legend_groupby_mapping = {str(k): str(v) for k, v in legend_groupby.items()}
                else:
                    raise TypeError("legend_groupby must be None, an obs column name, or a {category: group} mapping.")

            _add_categorical_legend(
                ax,
                color_source_vector,
                palette=_get_palette(adata, value_to_plot),
                scatter_array=coords,
                legend_loc=legend_loc,
                legend_numbered=legend_numbered,
                legend_on_data_numbers=legend_on_data_numbers,
                legend_number_start=legend_number_start,
                legend_title=legend_title,
                legend_fontweight=legend_fontweight,
                legend_fontsize=legend_fontsize,
                legend_fontoutline=path_effect,
                na_color=na_color,
                na_in_legend=na_in_legend,
                multi_panel=bool(grid),
                legend_style=legend_style,
                legend_groupby_vector=legend_groupby_vector,
                legend_groupby_mapping=legend_groupby_mapping,
                legend_group_order=legend_group_order,
                legend_columns=legend_columns,
                legend_marker_map=legend_marker_map,
                legend_edge_palette=legend_edge_palette,
                legend_ncols=legend_ncols,
                legend_scale=legend_scale,
                legend_density=legend_density,
                legend_panel_width=legend_panel_width,
                legend_panel_pad=legend_panel_pad,
                legend_badge_edgewidth=legend_badge_edgewidth,
                legend_badge_edgecolor=legend_badge_edgecolor,
                legend_label_sep=legend_label_sep,
                legend_show_border=legend_show_border,
            )
        elif colorbar_loc is not None:

            if frameon=='small' or frameon==False:

                from matplotlib.ticker import MaxNLocator

                # Get main axis position
                pos = ax.get_position()

                cb_height = pos.height * colorbar_height_fraction
                cb_bottom = pos.y0
                cb_w = colorbar_width if colorbar_width is not None else pos.width * 0.035
                cb_gap = pos.width * (colorbar_pad if colorbar_pad is not None else 0.015)

                cax1 = pl.gcf().add_axes([pos.x1 + cb_gap, cb_bottom, cb_w, cb_height])
                cax1.set_zorder(10)

                cb = pl.colorbar(cax, cax=cax1, orientation="vertical")
                cb.locator = MaxNLocator(nbins=3, integer=True)
                cb.update_ticks()
                if legend_fontsize is not None:
                    cb.ax.tick_params(labelsize=legend_fontsize)

            else:
                pl.colorbar(
                    cax, ax=ax, pad=0.01, fraction=0.08, aspect=30, location=colorbar_loc
                )

    if return_fig is True:
        return fig
    axs = axs if grid else ax
    _utils.savefig_or_show(basis, show=show, save=save)
    if show is False:
        return axs


def _panel_grid(hspace, wspace, ncols, num_panels):
    from matplotlib import gridspec

    n_panels_x = min(ncols, num_panels)
    n_panels_y = np.ceil(num_panels / n_panels_x).astype(int)
    # each panel will have the size of rcParams['figure.figsize']
    fig = pl.figure(
        figsize=(
            n_panels_x * rcParams['figure.figsize'][0] * (1 + wspace),
            n_panels_y * rcParams['figure.figsize'][1],
        ),
    )
    left = 0.2 / n_panels_x
    bottom = 0.13 / n_panels_y
    gs = gridspec.GridSpec(
        nrows=n_panels_y,
        ncols=n_panels_x,
        left=left,
        right=1 - (n_panels_x - 1) * left - 0.01 / n_panels_x,
        bottom=bottom,
        top=1 - (n_panels_y - 1) * bottom - 0.1 / n_panels_y,
        hspace=hspace,
        wspace=wspace,
    )
    return fig, gs


def _get_vboundnorm(
    vmin: Sequence[VBound],
    vmax: Sequence[VBound],
    vcenter: Sequence[VBound],
    norm: Sequence[Normalize],
    index: int,
    color_vector: Sequence[float],
) -> Tuple[Union[float, None], Union[float, None]]:
    """
    Evaluates the value of vmin, vmax and vcenter, which could be a
    str in which case is interpreted as a percentile and should
    be specified in the form 'pN' where N is the percentile.
    Eg. for a percentile of 85 the format would be 'p85'.
    Floats are accepted as p99.9

    Alternatively, vmin/vmax could be a function that is applied to
    the list of color values (`color_vector`).  E.g.

    def my_vmax(color_vector): np.percentile(color_vector, p=80)


    Parameters
    ----------
    index
        This index of the plot
    color_vector
        List or values for the plot

    Returns
    -------

    (vmin, vmax, vcenter, norm) containing None or float values for
    vmin, vmax, vcenter and matplotlib.colors.Normalize  or None for norm.

    """
    out = []
    for v_name, v in [('vmin', vmin), ('vmax', vmax), ('vcenter', vcenter)]:
        if len(v) == 1:
            # this case usually happens when the user sets eg vmax=0.9, which
            # is internally converted into list of len=1, but is expected that this
            # value applies to all plots.
            v_value = v[0]
        else:
            try:
                v_value = v[index]
            except IndexError:
                logg.error(
                    f"The parameter {v_name} is not valid. If setting multiple {v_name} values,"
                    f"check that the length of the {v_name} list is equal to the number "
                    "of plots. "
                )
                v_value = None

        if v_value is not None:
            if isinstance(v_value, str) and v_value.startswith('p'):
                try:
                    float(v_value[1:])
                except ValueError:
                    logg.error(
                        f"The parameter {v_name}={v_value} for plot number {index + 1} is not valid. "
                        f"Please check the correct format for percentiles."
                    )
                # interpret value of vmin/vmax as quantile with the following syntax 'p99.9'
                v_value = np.nanpercentile(color_vector, q=float(v_value[1:]))
            elif callable(v_value):
                # interpret vmin/vmax as function
                v_value = v_value(color_vector)
                if not isinstance(v_value, float):
                    logg.error(
                        f"The return of the function given for {v_name} is not valid. "
                        "Please check that the function returns a number."
                    )
                    v_value = None
            else:
                try:
                    float(v_value)
                except ValueError:
                    logg.error(
                        f"The given {v_name}={v_value} for plot number {index + 1} is not valid. "
                        f"Please check that the value given is a valid number, a string "
                        f"starting with 'p' for percentiles or a valid function."
                    )
                    v_value = None
        out.append(v_value)
    out.append(norm[0] if len(norm) == 1 else norm[index])
    return tuple(out)


def _wraps_plot_scatter(wrapper):
    import inspect

    params = inspect.signature(embedding).parameters.copy()
    wrapper_sig = inspect.signature(wrapper)
    wrapper_params = wrapper_sig.parameters.copy()

    params.pop("basis")
    params.pop("kwargs")
    wrapper_params.pop("adata")

    params.update(wrapper_params)
    annotations = {
        k: v.annotation
        for k, v in params.items()
        if v.annotation != inspect.Parameter.empty
    }
    if wrapper_sig.return_annotation is not inspect.Signature.empty:
        annotations["return"] = wrapper_sig.return_annotation

    wrapper.__signature__ = inspect.Signature(
        list(params.values()), return_annotation=wrapper_sig.return_annotation
    )
    wrapper.__annotations__ = annotations

    return wrapper


# API


@_wraps_plot_scatter
@_doc_params(
    adata_color_etc=doc_adata_color_etc,
    edges_arrows=doc_edges_arrows,
    scatter_bulk=doc_scatter_embedding,
    show_save_ax=doc_show_save_ax,
)
def umap(adata, **kwargs) -> Union[Axes, List[Axes], None]:
    r"""Scatter plot in UMAP basis.

    Arguments:
        adata: Annotated data matrix
        **kwargs: Additional arguments passed to embedding function

    Returns:
        Matplotlib axes or list of axes if show=False
    """
    return embedding(adata, 'umap', **kwargs)


@_wraps_plot_scatter
@_doc_params(
    adata_color_etc=doc_adata_color_etc,
    edges_arrows=doc_edges_arrows,
    scatter_bulk=doc_scatter_embedding,
    show_save_ax=doc_show_save_ax,
)
def tsne(adata, **kwargs) -> Union[Axes, List[Axes], None]:
    r"""Scatter plot in tSNE basis.

    Arguments:
        adata: Annotated data matrix
        **kwargs: Additional arguments passed to embedding function

    Returns:
        Matplotlib axes or list of axes if show=False
    """
    return embedding(adata, 'tsne', **kwargs)


@_wraps_plot_scatter
@_doc_params(
    adata_color_etc=doc_adata_color_etc,
    scatter_bulk=doc_scatter_embedding,
    show_save_ax=doc_show_save_ax,
)
def diffmap(adata, **kwargs) -> Union[Axes, List[Axes], None]:
    """\
    Scatter plot in Diffusion Map basis.

    Parameters
    ----------
    {adata_color_etc}
    {scatter_bulk}
    {show_save_ax}

    Returns
    -------
    If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.

    Examples
    --------
    .. plot::
        :context: close-figs

        import scanpy as sc
        adata = sc.datasets.pbmc68k_reduced()
        sc.tl.diffmap(adata)
        sc.pl.diffmap(adata, color='bulk_labels')

    .. currentmodule:: scanpy

    See also
    --------
    tl.diffmap
    """
    return embedding(adata, 'diffmap', **kwargs)


@_wraps_plot_scatter
@_doc_params(
    adata_color_etc=doc_adata_color_etc,
    edges_arrows=doc_edges_arrows,
    scatter_bulk=doc_scatter_embedding,
    show_save_ax=doc_show_save_ax,
)
def draw_graph(
    adata: AnnData, *, layout = None, **kwargs
) -> Union[Axes, List[Axes], None]:
    """\
    Scatter plot in graph-drawing basis.

    Parameters
    ----------
    {adata_color_etc}
    layout
        One of the :func:`~scanpy.tl.draw_graph` layouts.
        By default, the last computed layout is used.
    {edges_arrows}
    {scatter_bulk}
    {show_save_ax}

    Returns
    -------
    If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.

    Examples
    --------
    .. plot::
        :context: close-figs

        import scanpy as sc
        adata = sc.datasets.pbmc68k_reduced()
        sc.tl.draw_graph(adata)
        sc.pl.draw_graph(adata, color=['phase', 'bulk_labels'])

    .. currentmodule:: scanpy

    See also
    --------
    tl.draw_graph
    """
    if layout is None:
        layout = str(adata.uns['draw_graph']['params']['layout'])
    basis = 'draw_graph_' + layout
    if 'X_' + basis not in adata.obsm_keys():
        raise ValueError(
            'Did not find {} in adata.obs. Did you compute layout {}?'.format(
                'draw_graph_' + layout, layout
            )
        )

    return embedding(adata, basis, **kwargs)


@_wraps_plot_scatter
@_doc_params(
    adata_color_etc=doc_adata_color_etc,
    scatter_bulk=doc_scatter_embedding,
    show_save_ax=doc_show_save_ax,
)
def pca(
    adata,
    *,
    annotate_var_explained: bool = False,
    show: Optional[bool] = None,
    return_fig: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
) -> Union[Axes, List[Axes], None]:
    r"""Scatter plot in PCA coordinates.

    Arguments:
        adata: Annotated data matrix
        annotate_var_explained: Annotate explained variance (False)
        show: Show the plot (None)
        return_fig: Return figure object (None)
        save: Save the plot (None)
        **kwargs: Additional arguments passed to embedding function

    Returns:
        Matplotlib axes or list of axes if show=False
    """
    if not annotate_var_explained:
        return embedding(
            adata, 'pca', show=show, return_fig=return_fig, save=save, **kwargs
        )
    else:
        if 'pca' not in adata.obsm.keys() and 'X_pca' not in adata.obsm.keys():
            raise KeyError(
                f"Could not find entry in `obsm` for 'pca'.\n"
                f"Available keys are: {list(adata.obsm.keys())}."
            )

        label_dict = {
            'PC{}'.format(i + 1): 'PC{} ({}%)'.format(i + 1, round(v * 100, 2))
            for i, v in enumerate(adata.uns['pca']['variance_ratio'])
        }

        if return_fig is True:
            # edit axis labels in returned figure
            fig = embedding(adata, 'pca', return_fig=return_fig, **kwargs)
            for ax in fig.axes:
                ax.set_xlabel(label_dict[ax.xaxis.get_label().get_text()])
                ax.set_ylabel(label_dict[ax.yaxis.get_label().get_text()])
            return fig

        else:
            # get the axs, edit the labels and apply show and save from user
            axs = embedding(adata, 'pca', show=False, save=False, **kwargs)
            if isinstance(axs, list):
                for ax in axs:
                    ax.set_xlabel(label_dict[ax.xaxis.get_label().get_text()])
                    ax.set_ylabel(label_dict[ax.yaxis.get_label().get_text()])
            else:
                axs.set_xlabel(label_dict[axs.xaxis.get_label().get_text()])
                axs.set_ylabel(label_dict[axs.yaxis.get_label().get_text()])
            _utils.savefig_or_show('pca', show=show, save=save)
            if show is False:
                return axs


@_wraps_plot_scatter
@_doc_params(
    adata_color_etc=doc_adata_color_etc,
    scatter_spatial=doc_scatter_spatial,
    scatter_bulk=doc_scatter_embedding,
    show_save_ax=doc_show_save_ax,
)
def spatial(
    adata,
    *,
    basis: str = "spatial",
    img: Union[np.ndarray, None] = None,
    img_key: Union[str, None, Empty] = _empty,
    library_id: Union[str, None, Empty] = _empty,
    crop_coord: Tuple[int, int, int, int] = None,
    alpha_img: float = 1.0,
    bw: Optional[bool] = False,
    size: float = 1.0,
    scale_factor: Optional[float] = None,
    spot_size: Optional[float] = None,
    na_color: Optional[ColorLike] = None,
    show: Optional[bool] = None,
    return_fig: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
) -> Union[Axes, List[Axes], None]:
    """\
    Scatter plot in spatial coordinates.

    This function allows overlaying data on top of images.
    Use the parameter `img_key` to see the image in the background
    And the parameter `library_id` to select the image.
    By default, `'hires'` and `'lowres'` are attempted.

    Use `crop_coord`, `alpha_img`, and `bw` to control how it is displayed.
    Use `size` to scale the size of the Visium spots plotted on top.

    As this function is designed to for imaging data, there are two key assumptions
    about how coordinates are handled:

    1. The origin (e.g `(0, 0)`) is at the top left – as is common convention
    with image data.

    2. Coordinates are in the pixel space of the source image, so an equal
    aspect ratio is assumed.

    If your anndata object has a `"spatial"` entry in `.uns`, the `img_key`
    and `library_id` parameters to find values for `img`, `scale_factor`,
    and `spot_size` arguments. Alternatively, these values be passed directly.

    Parameters
    ----------
    {adata_color_etc}
    {scatter_spatial}
    {scatter_bulk}
    {show_save_ax}

    Returns
    -------
    If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.

    Examples
    --------
    This function behaves very similarly to other embedding plots like
    :func:`~scanpy.pl.umap`

    >>> adata = sc.datasets.visium_sge("Targeted_Visium_Human_Glioblastoma_Pan_Cancer")
    >>> sc.pp.calculate_qc_metrics(adata, inplace=True)
    >>> sc.pl.spatial(adata, color="log1p_n_genes_by_counts")

    See Also
    --------
    :func:`scanpy.datasets.visium_sge`
        Example visium data.
    :tutorial:`spatial/basic-analysis`
        Tutorial on spatial analysis.
    """
    # get default image params if available
    library_id, spatial_data = _check_spatial_data(adata.uns, library_id)
    img, img_key = _check_img(spatial_data, img, img_key, bw=bw)
    spot_size = _check_spot_size(spatial_data, spot_size)
    scale_factor = _check_scale_factor(
        spatial_data, img_key=img_key, scale_factor=scale_factor
    )
    crop_coord = _check_crop_coord(crop_coord, scale_factor)
    na_color = _check_na_color(na_color, img=img)

    if bw:
        cmap_img = "gray"
    else:
        cmap_img = None
    circle_radius = size * scale_factor * spot_size * 0.5

    axs = embedding(
        adata,
        basis=basis,
        scale_factor=scale_factor,
        size=circle_radius,
        na_color=na_color,
        show=False,
        save=False,
        **kwargs,
    )
    if not isinstance(axs, list):
        axs = [axs]
    for ax in axs:
        cur_coords = np.concatenate([ax.get_xlim(), ax.get_ylim()])
        if img is not None:
            ax.imshow(img, cmap=cmap_img, alpha=alpha_img)
        else:
            ax.set_aspect("equal")
            ax.invert_yaxis()
        if crop_coord is not None:
            ax.set_xlim(crop_coord[0], crop_coord[1])
            ax.set_ylim(crop_coord[3], crop_coord[2])
        else:
            ax.set_xlim(cur_coords[0], cur_coords[1])
            ax.set_ylim(cur_coords[3], cur_coords[2])
    _utils.savefig_or_show('show', show=show, save=save)
    if show is False or return_fig is True:
        return axs


# Helpers
def _components_to_dimensions(
    components: Optional[Union[str, Collection[str]]],
    dimensions: Optional[Union[Collection[int], Collection[Collection[int]]]],
    *,
    projection: Literal["2d", "3d"] = "2d",
    total_dims: int,
) -> List[Collection[int]]:
    """Normalize components/ dimensions args for embedding plots."""
    # TODO: Deprecate components kwarg
    ndims = {"2d": 2, "3d": 3}[projection]
    if components is None and dimensions is None:
        dimensions = [tuple(i for i in range(ndims))]
    elif components is not None and dimensions is not None:
        raise ValueError("Cannot provide both dimensions and components")

    # TODO: Consider deprecating this
    # If components is not None, parse them and set dimensions
    if components == "all":
        dimensions = list(combinations(range(total_dims), ndims))
    elif components is not None:
        if isinstance(components, str):
            components = [components]
        # Components use 1 based indexing
        dimensions = [[int(dim) - 1 for dim in c.split(",")] for c in components]

    if all(isinstance(el, Integral) for el in dimensions):
        dimensions = [dimensions]
    # if all(isinstance(el, Collection) for el in dimensions):
    for dims in dimensions:
        if len(dims) != ndims or not all(isinstance(d, Integral) for d in dims):
            raise ValueError()

    return dimensions



# -----------------------------------------------------------------------------
# Custom grouped categorical legends
# -----------------------------------------------------------------------------

def _legend_str_key(x):
    """Stable string key for category-based style dictionaries."""
    return str(x)


def _as_object_array(values):
    """Convert pandas/polars/categorical-like values to a numpy object array."""
    if values is None:
        return None
    if isinstance(values, pd.Categorical):
        return np.asarray(values.astype(object), dtype=object)
    if hasattr(values, "to_numpy"):
        try:
            return np.asarray(values.to_numpy(), dtype=object)
        except Exception:
            pass
    return np.asarray(values, dtype=object)


def _resolve_style_value(mapping, cat, default=None):
    """Resolve a style value from a mapping with either raw or string category keys."""
    if mapping is None:
        return default
    try:
        if cat in mapping:
            return mapping[cat]
    except Exception:
        pass
    return mapping.get(str(cat), default) if hasattr(mapping, "get") else default


def _infer_category_groups(cats_list, color_source_vector, legend_groupby_vector=None, legend_groupby_mapping=None):
    """
    Infer {category -> group} for the custom legend.

    Priority:
    1. explicit {category: group} mapping;
    2. paired per-cell legend_groupby_vector;
    3. a single empty group, meaning no visible group header.
    """
    out = {}
    if legend_groupby_mapping is not None:
        for cat in cats_list:
            out[cat] = str(_resolve_style_value(legend_groupby_mapping, cat, ""))
        return out

    if legend_groupby_vector is not None:
        cat_arr = _as_object_array(color_source_vector)
        grp_arr = _as_object_array(legend_groupby_vector)
        if cat_arr is not None and grp_arr is not None and len(cat_arr) == len(grp_arr):
            tmp = pd.DataFrame({"cat": [str(x) for x in cat_arr], "group": [str(x) for x in grp_arr]})
            tmp = tmp[~tmp["cat"].isin(["nan", "None"])]
            for cat in cats_list:
                sub = tmp.loc[tmp["cat"] == str(cat), "group"]
                if len(sub) == 0:
                    out[cat] = ""
                else:
                    # mode is more robust if a category appears with accidental mixed labels
                    out[cat] = str(sub.value_counts().index[0])
            return out

    return {cat: "" for cat in cats_list}


def _make_custom_legend_style_table(
    cats_list,
    palette,
    cat_to_number,
    legend_groupby_map,
    legend_group_order=None,
    legend_marker_map=None,
    legend_edge_palette=None,
    legend_badge_edgecolor=None,
):
    """Build a small table that drives the custom legend renderer."""
    rows = []
    for cat in cats_list:
        group = legend_groupby_map.get(cat, "")
        color = _resolve_style_value(palette, cat, "#808080")
        marker = _resolve_style_value(legend_marker_map, cat, "o")
        edge = _resolve_style_value(
            legend_edge_palette,
            cat,
            legend_badge_edgecolor if legend_badge_edgecolor is not None else "none",
        )
        rows.append(
            {
                "category": cat,
                "label": str(cat),
                "number": cat_to_number.get(cat, str(cat)),
                "group": str(group),
                "color": color,
                "edgecolor": edge,
                "marker": marker,
            }
        )
    style_df = pd.DataFrame(rows)

    if legend_group_order is not None:
        group_rank = {str(g): i for i, g in enumerate(legend_group_order)}
    else:
        group_rank = {g: i for i, g in enumerate(style_df["group"].drop_duplicates().tolist())}
    cat_rank = {cat: i for i, cat in enumerate(cats_list)}
    style_df["_group_rank"] = style_df["group"].map(lambda g: group_rank.get(str(g), 10_000))
    style_df["_cat_rank"] = style_df["category"].map(lambda c: cat_rank.get(c, 10_000))
    style_df = style_df.sort_values(["_group_rank", "_cat_rank"]).drop(columns=["_group_rank", "_cat_rank"])
    return style_df


def _split_groups_to_columns(style_df, legend_columns=None, legend_ncols=None):
    """Return list[list[group]] with semantic group blocks placed into columns."""
    groups = [str(g) for g in style_df["group"].drop_duplicates().tolist()]
    if legend_columns is not None:
        return [[str(g) for g in col] for col in legend_columns]

    # If no group headers are used, split category rows across ncols indirectly as one pseudo-group.
    if groups == [""]:
        return [[""]]

    if legend_ncols is None:
        n = len(style_df)
        legend_ncols = 1 if n <= 14 else 2 if n <= 30 else 3
    legend_ncols = max(int(legend_ncols), 1)
    cols = [[] for _ in range(legend_ncols)]
    heights = np.zeros(legend_ncols)
    for g in groups:
        n_items = int((style_df["group"] == g).sum())
        k = int(np.argmin(heights))
        cols[k].append(g)
        heights[k] += n_items + 1.7
    return cols


def _draw_round_badge(ax, x, y, w, h, facecolor, *, radius=0.0065, edgecolor="none", linewidth=0.0):
    """Draw a rounded rectangle/square in the legend axis [0,1] coordinate system."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.002,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transData,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def _layout_from_knobs(fontsize, scale, density, has_title):
    """Derive absolute legend geometry from the three public knobs.

    Font hierarchy (relative to item labels):
      title > group header > item label ≈ badge number

    Vertical rhythm (independent gaps):
      title → first group header
      group header → first item
      item → item
      last item → next group header
    """
    base_fs = float(fontsize) if fontsize is not None else 11.3
    scale = max(float(scale), 1e-6)
    density = max(float(density), 1e-6)
    item_fs = base_fs * scale
    # Larger default badge so numbers stay readable in publication figures
    badge_size = 0.054 * scale
    return {
        "item_fs": item_fs,
        # title > group > item
        "title_fs": item_fs * 1.35,
        "group_fs": item_fs * 1.15,
        # badge number ≈ item label size (fills the larger square)
        "num_fs": item_fs * 1.15,
        "badge_size": badge_size,
        "badge_radius": badge_size * 0.27,
        "badge_label_gap": badge_size * 0.40,
        "marker_size": 110.0 * (scale ** 2),
        # item-to-item and header-to-first-item: roomier
        "item_step": 0.052 * scale / density,
        "header_to_items": 0.07 * scale / density,
        # last-item → next group header (NOT stacked on a trailing item_step)
        "group_block_gap": 0.05 * scale / density,
        "x_margin": float(np.clip(0.018 * scale, 0.008, 0.04)),
        "col_gap": float(np.clip(0.008 * scale, 0.004, 0.03)),
        "title_y": 0.995,
        # pack legend content toward the top of the panel
        "start_y": 0.95 if has_title else 0.98,
        "bottom_pad": 0.02,
        # Horizontal offsets relative to badge/marker size (axes fraction).
        "marker_x": badge_size * 0.42,
        "marker_label_x": badge_size * 1.35,
        "shape_label_x": badge_size * 1.42,
    }


def _column_needed_height(style_df, groups, layout, show_headers):
    """Estimate vertical cursor travel for one legend column."""
    height = 0.0
    active = []
    for group in groups:
        sub = style_df[style_df["group"] == group]
        if len(sub) == 0:
            continue
        active.append(int(len(sub)))
    if not active:
        return 0.0
    for gi, n_items in enumerate(active):
        if show_headers:
            height += layout["header_to_items"]
        if n_items > 1:
            height += (n_items - 1) * layout["item_step"]
        if show_headers and gi < len(active) - 1:
            height += layout["group_block_gap"]
    return height


def _resolve_legend_layout(
    style_df,
    columns,
    *,
    fontsize=None,
    scale=1.0,
    density=1.0,
    has_title=False,
):
    """
    Resolve legend layout from fontsize / scale / density, then auto-fit to height.

    Fitting order:
    1. compress vertical spacing (raise effective density, up to 2.5x);
    2. if still overflowing, shrink scale (fonts/badges) down to item_fs >= 7;
    3. final spacing compress if still needed at the floor scale.
    """
    scale = max(float(scale), 1e-6)
    density = max(float(density), 1e-6)
    base_fs = float(fontsize) if fontsize is not None else 11.3
    min_scale = min(scale, 7.0 / base_fs) if base_fs > 0 else scale * 0.5
    max_density_mult = 2.5

    show_headers = style_df["group"].drop_duplicates().tolist() != [""]

    def _needed(layout):
        heights = []
        for groups in columns:
            use_groups = groups if show_headers else [""]
            heights.append(_column_needed_height(style_df, use_groups, layout, show_headers))
        return max(heights) if heights else 0.0

    layout = _layout_from_knobs(base_fs, scale, density, has_title)
    available = layout["start_y"] - layout["bottom_pad"]
    needed = _needed(layout)

    if needed > available and needed > 0 and available > 0:
        dens_boost = needed / available
        if dens_boost <= max_density_mult:
            layout = _layout_from_knobs(base_fs, scale, density * dens_boost, has_title)
        else:
            layout = _layout_from_knobs(base_fs, scale, density * max_density_mult, has_title)
            needed = _needed(layout)
            if needed > available and scale > min_scale:
                scale_factor = available / needed
                new_scale = max(min_scale, scale * scale_factor)
                layout = _layout_from_knobs(base_fs, new_scale, density * max_density_mult, has_title)
                needed = _needed(layout)
                if needed > available and needed > 0:
                    layout = _layout_from_knobs(
                        base_fs,
                        new_scale,
                        density * max_density_mult * (needed / available),
                        has_title,
                    )

    layout["show_headers"] = show_headers
    return layout


def _draw_custom_grouped_legend(
    legend_ax,
    style_df,
    *,
    legend_style="grouped_roundsquare",
    legend_title=None,
    legend_columns=None,
    legend_ncols=None,
    legend_fontsize=None,
    legend_scale=1.0,
    legend_density=1.0,
    legend_badge_edgewidth=0.0,
    legend_label_sep=" | ",
    legend_show_border=False,
):
    """
    Draw a publication-style grouped legend.

    Supported legend_style values:
    - default/scanpy: handled outside this function;
    - grouped_dot: colored dot + "number | label";
    - grouped_roundsquare: equal-width/equal-height rounded badge with number;
    - grouped_roundrect: rounded rectangular badge with number;
    - grouped_square: square badge with number;
    - grouped_shape: marker-shaped badge with number.

    Layout is driven by legend_fontsize / legend_scale / legend_density only.
    This function intentionally does not touch the embedding scatter itself.
    """
    legend_ax.set_axis_off()
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)

    if legend_show_border:
        bg = FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            linewidth=0.8,
            edgecolor="#d0d0d0",
            facecolor="white",
            alpha=0.96,
            transform=legend_ax.transData,
            zorder=0,
        )
        legend_ax.add_patch(bg)

    columns = _split_groups_to_columns(style_df, legend_columns=legend_columns, legend_ncols=legend_ncols)
    layout = _resolve_legend_layout(
        style_df,
        columns,
        fontsize=legend_fontsize,
        scale=legend_scale,
        density=legend_density,
        has_title=legend_title is not None,
    )
    show_group_headers = layout["show_headers"]

    if legend_title is not None:
        legend_ax.text(
            layout["x_margin"],
            layout["title_y"],
            str(legend_title),
            ha="left",
            va="top",
            fontsize=layout["title_fs"],
            fontweight="bold",
        )

    ncols = max(len(columns), 1)
    col_width = (1 - 2 * layout["x_margin"] - layout["col_gap"] * (ncols - 1)) / ncols
    item_fs = layout["item_fs"]
    num_fs = layout["num_fs"]
    badge_size = layout["badge_size"]
    badge_radius = layout["badge_radius"]
    badge_label_gap = layout["badge_label_gap"]
    group_header_texts = []

    for ci, groups in enumerate(columns):
        x0 = layout["x_margin"] + ci * (col_width + layout["col_gap"])
        y = layout["start_y"]
        if not show_group_headers:
            groups = [""]
        active_groups = [g for g in groups if len(style_df[style_df["group"] == g]) > 0]
        for gi, group in enumerate(active_groups):
            sub = style_df[style_df["group"] == group]

            if show_group_headers:
                group_text = legend_ax.text(
                    x0,
                    y,
                    str(group),
                    ha="left",
                    va="top",
                    fontsize=layout["group_fs"],
                    fontweight="bold",
                )
                group_header_texts.append(group_text)
                y -= layout["header_to_items"]

            rows = list(sub.iterrows())
            for i, (_, row) in enumerate(rows):
                num = str(row["number"])
                lab = str(row["label"])
                col = row["color"]
                edge = row["edgecolor"]
                marker = row["marker"]

                if legend_style in ("grouped_dot", "dot"):
                    mx = x0 + layout["marker_x"]
                    my = y
                    legend_ax.scatter(
                        [mx],
                        [my],
                        s=layout["marker_size"],
                        marker="o",
                        c=[col],
                        edgecolors="none",
                        linewidths=0,
                        zorder=2,
                    )
                    legend_ax.text(
                        x0 + layout["marker_label_x"],
                        y,
                        f"{num}{legend_label_sep}{lab}",
                        ha="left",
                        va="center",
                        fontsize=item_fs,
                    )

                elif legend_style in ("grouped_shape", "shape"):
                    mx = x0 + layout["marker_x"]
                    my = y
                    legend_ax.scatter(
                        [mx],
                        [my],
                        s=layout["marker_size"],
                        marker=marker,
                        c=[col],
                        edgecolors=edge,
                        linewidths=legend_badge_edgewidth,
                        zorder=2,
                    )
                    legend_ax.text(
                        mx, my, num,
                        ha="center", va="center",
                        fontsize=num_fs, fontweight="normal", zorder=3,
                    )
                    legend_ax.text(
                        x0 + layout["shape_label_x"], y, lab,
                        ha="left", va="center", fontsize=item_fs,
                    )

                elif legend_style in ("grouped_square", "square"):
                    badge_w = badge_h = badge_size
                    bx = x0
                    by = y - badge_h / 2
                    _draw_round_badge(
                        legend_ax, bx, by, badge_w, badge_h, col,
                        radius=0.001, edgecolor=edge, linewidth=legend_badge_edgewidth,
                    )
                    legend_ax.text(
                        bx + badge_w / 2, y, num,
                        ha="center", va="center",
                        fontsize=num_fs, fontweight="normal", zorder=3,
                    )
                    legend_ax.text(
                        bx + badge_w + badge_label_gap, y, lab,
                        ha="left", va="center", fontsize=item_fs,
                    )

                elif legend_style in ("grouped_roundrect", "roundrect"):
                    badge_h = badge_size
                    badge_w = badge_size * 1.25
                    bx = x0
                    by = y - badge_h / 2
                    _draw_round_badge(
                        legend_ax, bx, by, badge_w, badge_h, col,
                        radius=badge_radius, edgecolor=edge, linewidth=legend_badge_edgewidth,
                    )
                    legend_ax.text(
                        bx + badge_w / 2, y, num,
                        ha="center", va="center",
                        fontsize=num_fs, fontweight="normal", zorder=3,
                    )
                    legend_ax.text(
                        bx + badge_w + badge_label_gap, y, lab,
                        ha="left", va="center", fontsize=item_fs,
                    )

                elif legend_style in ("grouped_roundsquare", "roundsquare"):
                    badge_w = badge_h = badge_size
                    bx = x0
                    by = y - badge_h / 2
                    _draw_round_badge(
                        legend_ax, bx, by, badge_w, badge_h, col,
                        radius=badge_radius, edgecolor=edge, linewidth=legend_badge_edgewidth,
                    )
                    legend_ax.text(
                        bx + badge_w / 2, y, num,
                        ha="center", va="center",
                        fontsize=num_fs, fontweight="normal", zorder=3,
                    )
                    legend_ax.text(
                        bx + badge_w + badge_label_gap, y, lab,
                        ha="left", va="center", fontsize=item_fs,
                    )

                else:
                    raise ValueError(f"Unknown legend_style={legend_style!r}")

                # item-to-item only; group gap is applied separately below
                if i < len(rows) - 1:
                    y -= layout["item_step"]

            # last item → next group header (independent of item_step)
            if show_group_headers and gi < len(active_groups) - 1:
                y -= layout["group_block_gap"]

    if group_header_texts:
        legend_ax.figure.canvas.draw()
        renderer = legend_ax.figure.canvas.get_renderer()
        underline_pad = max(0.004, layout["badge_size"] * 0.12)
        for group_text in group_header_texts:
            bb = group_text.get_window_extent(renderer=renderer)
            bb_data = bb.transformed(legend_ax.transData.inverted())
            legend_ax.plot(
                [bb_data.x0, bb_data.x1],
                [bb_data.y0 - underline_pad, bb_data.y0 - underline_pad],
                color="black",
                linewidth=0.9,
                solid_capstyle="butt",
                clip_on=False,
                zorder=2,
            )


def _add_custom_right_margin_legend(ax, style_df, **kwargs):
    """Create a right-margin inset axis and draw the custom legend in it."""
    width = kwargs.pop("legend_panel_width", 0.92)
    xpad = kwargs.pop("legend_panel_pad", 0.035)
    legend_ax = ax.inset_axes([1.0 + xpad, 0.0, width, 1.0], transform=ax.transAxes)
    _draw_custom_grouped_legend(legend_ax, style_df, **kwargs)
    return legend_ax

def _add_categorical_legend(
    ax,
    color_source_vector,
    palette: dict,
    legend_loc: str,
    legend_numbered: bool,
    legend_on_data_numbers: bool,
    legend_number_start: int,
    legend_title: Optional[str],
    legend_fontweight,
    legend_fontsize,
    legend_fontoutline,
    multi_panel,
    na_color,
    na_in_legend: bool,
    scatter_array=None,
    legend_style: str = "default",
    legend_groupby_vector=None,
    legend_groupby_mapping: Optional[Mapping[str, str]] = None,
    legend_group_order: Optional[Sequence[str]] = None,
    legend_columns: Optional[Sequence[Sequence[str]]] = None,
    legend_marker_map: Optional[Mapping[str, str]] = None,
    legend_edge_palette: Optional[Mapping[str, ColorLike]] = None,
    legend_ncols: Optional[int] = None,
    legend_scale: float = 1.0,
    legend_density: float = 1.0,
    legend_panel_width: float = 0.92,
    legend_panel_pad: float = 0.035,
    legend_badge_edgewidth: float = 0.0,
    legend_badge_edgecolor: Optional[ColorLike] = None,
    legend_label_sep: str = " | ",
    legend_show_border: bool = False,
):
    """Add a legend to the passed Axes."""
    if na_in_legend and pd.isnull(color_source_vector).any():
        if "NA" in color_source_vector:
            raise NotImplementedError(
                "No fallback for null labels has been defined if NA already in categories."
            )
        # Ensure color_source_vector is categorical before adding categories
        if not hasattr(color_source_vector, 'add_categories'):
            color_source_vector = pd.Categorical(color_source_vector)
        color_source_vector = color_source_vector.add_categories("NA").fillna("NA")
        palette = palette.copy()
        palette["NA"] = na_color
    if color_source_vector.dtype == bool:
        cats = pd.Categorical(color_source_vector.astype(str)).categories
    else:
        # Safely get categories - handle both Categorical and Series objects
        if hasattr(color_source_vector, 'categories'):
            cats = color_source_vector.categories
        else:
            # Convert to categorical if it's not already
            cats = pd.Categorical(color_source_vector).categories

    if multi_panel is True:
        # Shrink current axis by 10% to fit legend and match
        # size of plots that are not categorical
        if legend_loc == 'right margin':
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.91, box.height])

    # Stable category order for numbering and legend display
    cats_list = list(cats)
    if legend_numbered:
        numbers = list(range(int(legend_number_start), int(legend_number_start) + len(cats_list)))
        cat_to_label = {cat: f"{num}  {cat}" for cat, num in zip(cats_list, numbers)}
        cat_to_number = {cat: str(num) for cat, num in zip(cats_list, numbers)}
    else:
        cat_to_label = {cat: str(cat) for cat in cats_list}
        cat_to_number = {cat: str(cat) for cat in cats_list}

    if legend_loc == 'right margin':
        use_custom_legend = legend_style not in (None, "default", "scanpy", "standard")
        if use_custom_legend:
            legend_groupby_map = _infer_category_groups(
                cats_list,
                color_source_vector,
                legend_groupby_vector=legend_groupby_vector,
                legend_groupby_mapping=legend_groupby_mapping,
            )
            style_df = _make_custom_legend_style_table(
                cats_list,
                palette,
                cat_to_number,
                legend_groupby_map,
                legend_group_order=legend_group_order,
                legend_marker_map=legend_marker_map,
                legend_edge_palette=legend_edge_palette,
                legend_badge_edgecolor=legend_badge_edgecolor,
            )
            _add_custom_right_margin_legend(
                ax,
                style_df,
                legend_style=legend_style,
                legend_title=legend_title,
                legend_columns=legend_columns,
                legend_ncols=legend_ncols,
                legend_fontsize=legend_fontsize,
                legend_scale=legend_scale,
                legend_density=legend_density,
                legend_panel_width=legend_panel_width,
                legend_panel_pad=legend_panel_pad,
                legend_badge_edgewidth=legend_badge_edgewidth,
                legend_label_sep=legend_label_sep,
                legend_show_border=legend_show_border,
            )
        else:
            for cat in cats_list:
                ax.scatter([], [], c=palette[cat], label=cat_to_label[cat])
            ax.legend(
                frameon=False,
                loc='center left',
                bbox_to_anchor=(1, 0.5),
                ncol=(1 if len(cats) <= 14 else 2 if len(cats) <= 30 else 3),
                fontsize=legend_fontsize,
                title=legend_title,
            )
            if legend_title is not None:
                leg = ax.get_legend()
                if leg is not None:
                    leg.get_title().set_fontweight(legend_fontweight)
                    if legend_fontsize is not None:
                        leg.get_title().set_fontsize(legend_fontsize)

    if legend_loc == 'on data' or legend_on_data_numbers:
        # identify centroids to put labels

        # Use a category array without index alignment to keep coords and labels in sync.
        if isinstance(color_source_vector, pd.Categorical):
            groupby_key = color_source_vector
        else:
            groupby_key = pd.Categorical(np.asarray(color_source_vector))

        all_pos = (
            pd.DataFrame(scatter_array, columns=["x", "y"])
            .groupby(groupby_key, observed=True)
            .median()
            # Have to sort_index since if observed=True and categorical is unordered
            # the order of values in .index is undefined. Related issue:
            # https://github.com/pandas-dev/pandas/issues/25167
            .sort_index()
        )

        # Convert legend_fontoutline to PathEffect list if needed
        if legend_fontoutline is not None:
            if isinstance(legend_fontoutline, (int, float)):
                # Create white stroke outline with specified width
                text_path_effects = [
                    patheffects.withStroke(linewidth=legend_fontoutline, foreground='white')
                ]
            elif isinstance(legend_fontoutline, list):
                # Already a list of PathEffects
                text_path_effects = legend_fontoutline
            else:
                # Single PathEffect object
                text_path_effects = [legend_fontoutline]
        else:
            text_path_effects = None

        # If we're drawing labels on data, either write the original label, or the numeric index.
        # Scale with legend_scale; numbered on-data labels default a bit larger than legend items.
        base_fs = float(legend_fontsize) if legend_fontsize is not None else 11.3
        on_data_fs = base_fs * float(legend_scale) * 1.45
        for label, x_pos, y_pos in all_pos.itertuples():
            ax.text(
                x_pos,
                y_pos,
                cat_to_number.get(label, str(label)) if (legend_numbered or legend_on_data_numbers) else str(label),
                weight=legend_fontweight,
                verticalalignment='center',
                horizontalalignment='center',
                fontsize=on_data_fs,
                path_effects=text_path_effects,
            )


def embedding_numbered(
    adata: AnnData,
    basis: str,
    *,
    color: Union[str, Sequence[str], None] = None,
    legend_title: Optional[str] = None,
    legend_number_start: int = 1,
    legend_style: str = "default",
    legend_groupby: Optional[Union[str, Mapping[str, str]]] = None,
    legend_group_order: Optional[Sequence[str]] = None,
    legend_columns: Optional[Sequence[Sequence[str]]] = None,
    legend_marker_map: Optional[Mapping[str, str]] = None,
    legend_edge_palette: Optional[Mapping[str, ColorLike]] = None,
    legend_ncols: Optional[int] = None,
    legend_scale: float = 1.0,
    legend_density: float = 1.0,
    legend_panel_width: float = 0.92,
    legend_panel_pad: float = 0.035,
    legend_badge_edgewidth: float = 0.0,
    legend_badge_edgecolor: Optional[ColorLike] = None,
    **kwargs,
) -> Union[Figure, Axes, None]:
    """
    A derived version of `embedding()` that:
    - writes numeric indices (1..N) on the embedding (at per-group centroids),
    - and shows a right-side legend with `"<idx>  <group>"` labels.

    Notes:
    - Intended for categorical `color` (i.e. adata.obs group labels).
    - `legend_title` controls the legend title (e.g. "Cell type").
    - Legend layout is controlled by `legend_fontsize` / `legend_scale` / `legend_density`.
    """
    return embedding(
        adata=adata,
        basis=basis,
        color=color,
        legend_loc="right margin",
        legend_numbered=True,
        legend_on_data_numbers=True,
        legend_number_start=legend_number_start,
        legend_title=legend_title,
        legend_style=legend_style,
        legend_groupby=legend_groupby,
        legend_group_order=legend_group_order,
        legend_columns=legend_columns,
        legend_marker_map=legend_marker_map,
        legend_edge_palette=legend_edge_palette,
        legend_ncols=legend_ncols,
        legend_scale=legend_scale,
        legend_density=legend_density,
        legend_panel_width=legend_panel_width,
        legend_panel_pad=legend_panel_pad,
        legend_badge_edgewidth=legend_badge_edgewidth,
        legend_badge_edgecolor=legend_badge_edgecolor,
        **kwargs,
    )


def _get_basis(adata: AnnData, basis: str) -> np.ndarray:
    """Get array for basis from anndata. Just tries to add 'X_'."""
    if basis in adata.obsm:
        return adata.obsm[basis]
    elif f"X_{basis}" in adata.obsm:
        return adata.obsm[f"X_{basis}"]
    else:
        raise KeyError(f"Could not find '{basis}' or 'X_{basis}' in .obsm")


def _safe_check_obs_columns(adata, key):
    """Safely check if a key exists in adata.obs, compatible with both pandas and Rust backends."""
    try:
        # For pandas DataFrame
        if hasattr(adata.obs, 'columns'):
            return key in adata.obs.columns
        # For Rust/Polars backends that don't have .columns attribute
        else:
            # Try to access the column directly - if it exists, this won't raise an error
            try:
                _ = adata.obs[key]
                return True
            except (KeyError, IndexError):
                return False
    except Exception:
        return False

def _safe_check_var_names(adata, key):
    """Safely check if a key exists in adata.var_names, compatible with both pandas and Rust backends."""
    try:
        return key in adata.var_names
    except Exception:
        # Fallback for Rust backends
        try:
            if hasattr(adata.var_names, '__contains__'):
                return key in adata.var_names
            else:
                # Convert to list and check
                return key in list(adata.var_names)
        except Exception:
            return False

def _get_color_source_vector(
    adata, value_to_plot, use_raw=False, gene_symbols=None, layer=None, groups=None
):
    """
    Get array from adata that colors will be based on.
    Compatible with both pandas and Rust/Polars anndata backends.
    """
    if value_to_plot is None:
        # Points will be plotted with `na_color`. Ideally this would work
        # with the "bad color" in a color map but that throws a warning. Instead
        # _color_vector handles this.
        # https://github.com/matplotlib/matplotlib/issues/18294
        return np.broadcast_to(np.nan, adata.n_obs)

    # Safe checks for obs and var
    in_obs = _safe_check_obs_columns(adata, value_to_plot)
    # When use_raw is True, check raw.var_names; otherwise check var_names
    if use_raw and adata.raw is not None:
        in_var = _safe_check_var_names(adata.raw, value_to_plot)
    else:
        in_var = _safe_check_var_names(adata, value_to_plot)

    # Handle gene symbols - convert to actual gene names if needed
    if (
        gene_symbols is not None
        and not in_obs
        and not in_var
    ):
        # We should probably just make an index for this, and share it over runs
        try:
            if use_raw and adata.raw is not None:
                value_to_plot = adata.raw.var.index[adata.raw.var[gene_symbols] == value_to_plot][0]
                in_var = _safe_check_var_names(adata.raw, value_to_plot)  # Update after conversion
            else:
                value_to_plot = adata.var.index[adata.var[gene_symbols] == value_to_plot][0]
                in_var = _safe_check_var_names(adata, value_to_plot)  # Update after conversion
        except (IndexError, KeyError):
            pass  # Will be handled in the error case below

    # Determine the source of the data
    if in_obs:
        # Data is in adata.obs (metadata)
        values = adata.obs[value_to_plot]
    elif use_raw and in_var:
        # Data is gene expression from raw
        values = adata.raw.obs_vector(value_to_plot)
    elif in_var:
        # Data is gene expression from processed data
        values = adata.obs_vector(value_to_plot, layer=layer)
    else:
        # Last resort - try obs_vector which might handle other cases
        try:
            values = adata.obs_vector(value_to_plot, layer=layer)
        except (KeyError, AttributeError):
            raise KeyError(f"Could not find '{value_to_plot}' in adata.obs or adata.var_names")

    # Only convert string/object data to categorical, avoid converting numeric gene expression data
    if not isinstance(values.dtype, pd.CategoricalDtype):
        arr = np.asarray(values)
        # Only convert to categorical if data is string type and has duplicates
        if arr.dtype.kind in ("U", "S", "O"):  # string/object
            # Only convert if it's actually "categorical" rather than all unique
            if pd.unique(arr).size < arr.size:
                values = pd.Categorical(arr)
        # For numeric data (gene expression), keep as is, don't convert to categorical

    if groups and isinstance(values.dtype, pd.CategoricalDtype):
        values = values.remove_categories(values.categories.difference(groups))
    return values


def _get_palette(adata, values_key: str, palette=None):
    """
    Return {category -> hex}.
    - Python anndata: prioritize reading uns['<key>_colors'], fall back to '<key>_colors_rgba' if insufficient/missing.
    - Rust/Polars: only read uns['<key>_colors_rgba'] (avoid reading string arrays to prevent PanicException).
    - If none or insufficient length, generate default and write back:
        * Rust/Polars: only write '<key>_colors_rgba' (float32 RGBA)
        * Python anndata: write both '<key>_colors' (unicode) and '<key>_colors_rgba'
    """
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    from matplotlib import rcParams
    from matplotlib.colors import to_hex, to_rgba, is_color_like
    from cycler import Cycler

    color_key = f"{values_key}_colors"
    color_key_rgba = f"{values_key}_colors_rgba"

    # --------- Check if Rust/Polars backend (avoid reading string arrays) ---------
    def _is_rust_backend():
        try:
            if type(adata.obs).__name__.endswith("PyDataFrameElem"):
                return True
        except Exception:
            pass
        try:
            if type(adata.uns).__name__.endswith("PyElemCollection"):
                return True
        except Exception:
            pass
        # Fallback: module names containing snapatac2 / pyanndata
        m = type(adata).__module__
        return ("snapatac2" in m) or ("pyanndata" in m)

    IS_RUST = _is_rust_backend()

    # --------- Convert obs column to pandas.Categorical and get ordered categories ---------
    def _obs_to_categorical(adata, key):
        s = adata.obs[key]
        try:
            import polars as pl
        except Exception:
            pl = None

        if s.__class__.__module__.startswith("pandas"):
            if isinstance(s.dtype, pd.CategoricalDtype):
                cats = [str(x) for x in s.cat.categories]
                return pd.Categorical(pd.Series(s).astype(str), categories=cats)
            if getattr(s, "dtype", None) == bool:
                return pd.Categorical(pd.Series(s).astype(str))
            return pd.Categorical(pd.Series(s, dtype="string"))

        if pl is not None and isinstance(s, pl.Series):
            if s.dtype == pl.Boolean:
                return pd.Categorical(pd.Series(s.to_list()).astype(str), categories=["False", "True"])
            if s.dtype == pl.Categorical and hasattr(s.cat, "get_categories"):
                cats = [str(x) for x in s.cat.get_categories().to_list()]
                return pd.Categorical(pd.Series(s.to_list()).astype(str), categories=cats)
            arr = [str(x) for x in s.cast(pl.Utf8).to_list()]
            try:
                from natsort import natsorted
                cats = natsorted(pd.unique(pd.Series(arr)).tolist())
            except Exception:
                cats = sorted(pd.unique(pd.Series(arr)).tolist(), key=str)
            return pd.Categorical(arr, categories=cats)

        # Fallback
        arr = np.asarray(s, dtype=object)
        if arr.size and isinstance(arr.flat[0], (np.bool_, bool)):
            return pd.Categorical(pd.Series(arr).astype(str))
        return pd.Categorical([str(x) for x in arr])

    values = _obs_to_categorical(adata, values_key)
    cats = list(values.categories)
    n_cat = len(cats)

    # --------- Write colors (dual-track: Rust only RGBA; Python string+RGBA) ---------
    def _write_colors(hex_list):
        rgba = np.asarray([to_rgba(h) for h in hex_list], dtype=np.float32)
        adata.uns[color_key_rgba] = rgba
        if not IS_RUST:
            adata.uns[color_key] = np.asarray(hex_list, dtype="U16")

    # --------- Handle user-provided palette ---------
    if palette is not None:
        if isinstance(palette, dict):
            hex_list = [to_hex(palette.get(cat, "#808080"), keep_alpha=True) for cat in cats]
            _write_colors(hex_list)
            return dict(zip(cats, hex_list))

        if isinstance(palette, str) and (palette in mpl.colormaps):
            cmap = mpl.colormaps[palette]
            denom = max(n_cat - 1, 1)
            hex_list = [to_hex(cmap(i/denom), keep_alpha=True) for i in range(n_cat)]
            _write_colors(hex_list)
            return dict(zip(cats, hex_list))

        if isinstance(palette, (list, tuple)):
            try:
                from scanpy.plotting._utils import additional_colors
            except Exception:
                additional_colors = {}
            try:
                seq = [(c if is_color_like(c) else additional_colors[c]) for c in palette]
            except Exception as e:
                raise ValueError(f"Invalid color in palette: {e}") from None
            hex_list = [to_hex(seq[i % len(seq)], keep_alpha=True) for i in range(n_cat)]
            _write_colors(hex_list)
            return dict(zip(cats, hex_list))

        if isinstance(palette, Cycler):
            cc = palette()
            hex_list = [to_hex(next(cc)["color"], keep_alpha=True) for _ in range(n_cat)]
            _write_colors(hex_list)
            return dict(zip(cats, hex_list))

        raise ValueError(
            "palette must be a matplotlib colormap name, a sequence of colors, "
            "a dict {category: color}, or a cycler(color=...)."
        )

    # --------- No palette provided: try to read existing colors ---------
    hex_list = None
    if IS_RUST:
        # Rust: only read RGBA; don't touch '<key>_colors', even with try it will bubble up
        try:
            v = adata.uns[color_key_rgba]
            arr = v.to_numpy() if hasattr(v, "to_numpy") else np.asarray(v)
            if arr.ndim == 2 and arr.shape[1] in (3,4):
                hex_list = [to_hex(tuple(row), keep_alpha=True) for row in arr]
        except BaseException:  # Note: PanicException may not be Exception
            hex_list = None
    else:
        # Python: prioritize reading string arrays
        try:
            v = adata.uns[color_key]
            if hasattr(v, "to_list"):
                v = v.to_list()
            arr = np.asarray(v)
            if arr.dtype.kind in ("U","S","O"):
                hex_list = [str(x) for x in (arr.tolist() if isinstance(arr, np.ndarray) else list(arr))]
        except Exception:
            hex_list = None
        # Fall back to RGBA
        if hex_list is None:
            try:
                v = adata.uns[color_key_rgba]
                arr = v.to_numpy() if hasattr(v, "to_numpy") else np.asarray(v)
                if arr.ndim == 2 and arr.shape[1] in (3,4):
                    hex_list = [to_hex(tuple(row), keep_alpha=True) for row in arr]
            except Exception:
                hex_list = None

    # --------- If still none/insufficient length: generate default and write back ---------
    if (hex_list is None) or (len(hex_list) < n_cat):
        base = rcParams["axes.prop_cycle"].by_key().get("color", [])
        if len(base) >= n_cat:
            cc = rcParams["axes.prop_cycle"]()
            hex_list = [to_hex(next(cc)["color"], keep_alpha=True) for _ in range(n_cat)]
        else:
            try:
                from ..plot._palette import sc_color, palette_56, palette_112
            except Exception:
                sc_color = palette_56 = palette_112 = None
            if sc_color is not None and n_cat <= len(sc_color):
                hex_list = [to_hex(c, keep_alpha=True) for c in sc_color[:n_cat]]
            elif palette_56 is not None and n_cat <= 56:
                hex_list = [to_hex(c, keep_alpha=True) for c in palette_56[:n_cat]]
            elif palette_112 is not None and n_cat <= 112:
                hex_list = [to_hex(c, keep_alpha=True) for c in palette_112[:n_cat]]
            else:
                hex_list = ["#808080"] * n_cat
                try:
                    from scanpy import logging as logg
                    logg.info(
                        f"the obs value {values_key!r} has many categories; using uniform grey."
                    )
                except Exception:
                    pass
        _write_colors(hex_list)

    return dict(zip(cats, hex_list))



def _color_vector(
    adata, values_key: str, values, palette, na_color="lightgray"
) -> Tuple[np.ndarray, bool]:
    """
    Map array of values to array of hex (plus alpha) codes.

    For categorical data, the return value is list of colors taken
    from the category palette or from the given `palette` value.

    For continuous values, the input array is returned (may change in future).
    """
    ###
    # when plotting, the color of the dots is determined for each plot
    # the data is either categorical or continuous and the data could be in
    # 'obs' or in 'var'
    to_hex = partial(colors.to_hex, keep_alpha=True)
    if values_key is None:
        return np.broadcast_to(to_hex(na_color), adata.n_obs), False
    if isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == bool:
        if values.dtype == bool:
            values = pd.Categorical(values.astype(str))
        color_map = {
            k: to_hex(v)
            for k, v in _get_palette(adata, values_key, palette=palette).items()
        }
        # If color_map does not have unique values, this can be slow as the
        # result is not categorical
        color_vector = pd.Categorical(values.map(color_map))

        # Set color to 'missing color' for all missing values
        if color_vector.isna().any():
            color_vector = color_vector.add_categories([to_hex(na_color)])
            color_vector = color_vector.fillna(to_hex(na_color))
        return color_vector, True
    elif not isinstance(values.dtype, pd.CategoricalDtype):
        return values, False


def _basis2name(basis):
    """
    converts the 'basis' into the proper name.
    """
    key = basis[2:] if isinstance(basis, str) and basis.startswith("X_") else basis

    component_name = (
        'DC'
        if key == 'diffmap'
        else 't-SNE'
        if key == 'tsne'
        else 'UMAP'
        if key == 'umap'
        else 'PC'
        if key == 'pca'
        else key.replace('draw_graph_', '').upper()
        if 'draw_graph' in str(key)
        else str(key).upper() if str(key).islower() else key
    )
    return component_name


def _check_spot_size(
    spatial_data: Optional[Mapping], spot_size: Optional[float]
) -> float:
    """
    Resolve spot_size value.

    This is a required argument for spatial plots.
    """
    if spatial_data is None and spot_size is None:
        raise ValueError(
            "When .uns['spatial'][library_id] does not exist, spot_size must be "
            "provided directly."
        )
    elif spot_size is None:
        return spatial_data['scalefactors']['spot_diameter_fullres']
    else:
        return spot_size


def _check_scale_factor(
    spatial_data: Optional[Mapping],
    img_key: Optional[str],
    scale_factor: Optional[float],
) -> float:
    """Resolve scale_factor, defaults to 1."""
    if scale_factor is not None:
        return scale_factor
    elif spatial_data is not None and img_key is not None:
        return spatial_data['scalefactors'][f"tissue_{img_key}_scalef"]
    else:
        return 1.0


def _check_spatial_data(
    uns: Mapping, library_id: Union[str, None, Empty]
) -> Tuple[Optional[str], Optional[Mapping]]:
    """
    Given a mapping, try and extract a library id/ mapping with spatial data.

    Assumes this is `.uns` from how we parse visium data.
    """
    spatial_mapping = uns.get("spatial", {})
    if library_id is _empty:
        if len(spatial_mapping) > 1:
            raise ValueError(
                "Found multiple possible libraries in `.uns['spatial']. Please specify."
                f" Options are:\n\t{list(spatial_mapping.keys())}"
            )
        elif len(spatial_mapping) == 1:
            library_id = list(spatial_mapping.keys())[0]
        else:
            library_id = None
    if library_id is not None:
        spatial_data = spatial_mapping[library_id]
    else:
        spatial_data = None
    return library_id, spatial_data


def _check_img(
    spatial_data: Optional[Mapping],
    img: Optional[np.ndarray],
    img_key: Union[None, str, Empty],
    bw: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Resolve image for spatial plots.
    """
    if img is None and spatial_data is not None and img_key is _empty:
        img_key = next(
            (k for k in ['hires', 'lowres'] if k in spatial_data['images']),
        )  # Throws StopIteration Error if keys not present
    if img is None and spatial_data is not None and img_key is not None:
        img = spatial_data["images"][img_key]
    if bw:
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return img, img_key


def _check_crop_coord(
    crop_coord: Optional[tuple],
    scale_factor: float,
) -> Tuple[float, float, float, float]:
    """Handle cropping with image or basis."""
    if crop_coord is None:
        return None
    if len(crop_coord) != 4:
        raise ValueError("Invalid crop_coord of length {len(crop_coord)}(!=4)")
    crop_coord = tuple(c * scale_factor for c in crop_coord)
    return crop_coord


def _check_na_color(
    na_color: Optional[ColorLike], *, img: Optional[np.ndarray] = None
) -> ColorLike:
    if na_color is None:
        if img is not None:
            na_color = (0.0, 0.0, 0.0, 0.0)
        else:
            na_color = "lightgray"
    return na_color


def _broadcast_args(*args):
    """Broadcasts arguments to a common length."""
    from itertools import repeat

    lens = [len(arg) for arg in args]
    longest = max(lens)
    if not (set(lens) == {1, longest} or set(lens) == {longest}):
        raise ValueError(f"Could not broadast together arguments with shapes: {lens}.")
    return list(
        [[arg[0] for _ in range(longest)] if len(arg) == 1 else arg for arg in args]
    )

def _embedding(
    adata: AnnData,
    basis: str,
    *,
    color: Union[str, Sequence[str], None] = None,
    gene_symbols: Optional[str] = None,
    use_raw: Optional[bool] = None,
    sort_order: bool = True,
    edges: bool = False,
    edges_width: float = 0.1,
    edges_color: Union[str, Sequence[float], Sequence[str]] = 'grey',
    neighbors_key: Optional[str] = None,
    arrows: bool = False,
    arrows_kwds: Optional[Mapping[str, Any]] = None,
    groups: Optional[str] = None,
    components: Union[str, Sequence[str]] = None,
    dimensions: Optional[Union[Tuple[int, int], Sequence[Tuple[int, int]]]] = None,
    layer: Optional[str] = None,
    projection: Literal['2d', '3d'] = '2d',
    scale_factor: Optional[float] = None,
    color_map: Union[Colormap, str, None] = None,
    cmap: Union[Colormap, str, None] = None,
    palette: Union[str, Sequence[str], Cycler, None] = None,
    na_color: ColorLike = "lightgray",
    na_in_legend: bool = True,
    size: Union[float, Sequence[float], None] = None,
    frameon: Optional[bool] = None,
    legend_fontsize: Union[int, float, _FontSize, None] = None,
    legend_fontweight: Union[int, _FontWeight] = 'bold',
    legend_loc: str = 'right margin',
    legend_style: str = "default",
    legend_groupby: Optional[Union[str, Mapping[str, str]]] = None,
    legend_group_order: Optional[Sequence[str]] = None,
    legend_columns: Optional[Sequence[Sequence[str]]] = None,
    legend_marker_map: Optional[Mapping[str, str]] = None,
    legend_edge_palette: Optional[Mapping[str, ColorLike]] = None,
    legend_ncols: Optional[int] = None,
    legend_scale: float = 1.0,
    legend_density: float = 1.0,
    legend_panel_width: float = 0.92,
    legend_panel_pad: float = 0.035,
    legend_badge_edgewidth: float = 0.0,
    legend_badge_edgecolor: Optional[ColorLike] = None,
    legend_label_sep: str = " | ",
    legend_show_border: bool = False,
    legend_fontoutline: Optional[int] = None,
    colorbar_loc: Optional[str] = "right",
    colorbar_width: Optional[float] = None,
    colorbar_pad: Optional[float] = None,
    colorbar_height_fraction: float = 0.3,
    vmax: Union[VBound, Sequence[VBound], None] = None,
    vmin: Union[VBound, Sequence[VBound], None] = None,
    vcenter: Union[VBound, Sequence[VBound], None] = None,
    norm: Union[Normalize, Sequence[Normalize], None] = None,
    add_outline: Optional[bool] = False,
    outline_width: Tuple[float, float] = (0.3, 0.05),
    outline_color: Tuple[str, str] = ('black', 'white'),
    ncols: int = 4,
    hspace: float = 0.25,
    wspace: Optional[float] = None,
    title: Union[str, Sequence[str], None] = None,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    ax: Optional[Axes] = None,
    return_fig: Optional[bool] = None,
    marker: Union[str, Sequence[str]] = '.',
    **kwargs,
) -> Union[Figure, Axes, None]:
    """\
    Scatter plot for user specified embedding basis (e.g. umap, pca, etc)

    Arguments:
        adata: Annotated data matrix.
        basis: Name of the `obsm` basis to use.
        
    Returns:
        If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.
    """

    return embedding(adata=adata, basis=basis, color=color, 
                     gene_symbols=gene_symbols, use_raw=use_raw, 
                     sort_order=sort_order, edges=edges, 
                     edges_width=edges_width, edges_color=edges_color, 
                     neighbors_key=neighbors_key, arrows=arrows, 
                     arrows_kwds=arrows_kwds, groups=groups, 
                     components=components, dimensions=dimensions, 
                     layer=layer, projection=projection, scale_factor=scale_factor,
                       color_map=color_map, cmap=cmap, palette=palette, 
                       na_color=na_color, na_in_legend=na_in_legend, 
                       size=size, frameon=frameon, legend_fontsize=legend_fontsize, 
                       legend_fontweight=legend_fontweight, legend_loc=legend_loc, 
                       legend_style=legend_style, legend_groupby=legend_groupby,
                       legend_group_order=legend_group_order, legend_columns=legend_columns,
                       legend_marker_map=legend_marker_map, legend_edge_palette=legend_edge_palette,
                       legend_ncols=legend_ncols, legend_scale=legend_scale,
                       legend_density=legend_density, legend_panel_width=legend_panel_width,
                       legend_panel_pad=legend_panel_pad,
                       legend_badge_edgewidth=legend_badge_edgewidth,
                       legend_badge_edgecolor=legend_badge_edgecolor, legend_label_sep=legend_label_sep,
                       legend_show_border=legend_show_border,
                       legend_fontoutline=legend_fontoutline, colorbar_loc=colorbar_loc,
                       colorbar_width=colorbar_width, colorbar_pad=colorbar_pad,
                       colorbar_height_fraction=colorbar_height_fraction,
                       vmax=vmax, vmin=vmin, vcenter=vcenter, norm=norm,
                       add_outline=add_outline, outline_width=outline_width, 
                       outline_color=outline_color, ncols=ncols, hspace=hspace,
                         wspace=wspace, title=title, show=show, save=save, ax=ax,
                           return_fig=return_fig, marker=marker, **kwargs)


# === drop-in replacement: pandas / polars compatible ===


import numpy as np
import pandas as pd
from typing import Mapping, Sequence
from cycler import Cycler
import matplotlib as mpl
from matplotlib.colors import is_color_like, to_hex, to_rgba
from natsort import natsorted

# 可选：与 scanpy 的 warning 接口对齐
try:
    from scanpy import logging as logg
except Exception:
    class _LogStub:
        def warning(self, *a, **k): pass
    logg = _LogStub()

# 可选：scanpy 自带的颜色名扩展（没有也不影响）
try:
    from scanpy.plotting._utils import additional_colors  # type: ignore
except Exception:
    additional_colors: Mapping[str, str] = {}

def _obs_series(adata, key):
    """兼容 pandas/Polars 的 obs 列取值。"""
    try:
        return adata.obs[key]
    except Exception as e:
        raise KeyError(f"obs does not contain column {key!r}") from e

def _obs_categories_ordered(adata, key):
    """Get categorical categories (in existing order); if not categorical, get unique values and sort naturally."""
    s = _obs_series(adata, key)

    # pandas.Series
    if s.__class__.__module__.startswith("pandas"):
        if isinstance(s.dtype, pd.CategoricalDtype):
            cats = list(s.cat.categories)
        else:
            cats = list(pd.unique(pd.Series(s, dtype="string")))
            cats = [str(x) for x in cats]
            cats = natsorted(cats)
        return [str(x) for x in cats]

    # Polars.Series
    import polars as pl
    if pl is not None and isinstance(s, pl.Series):
        if s.dtype == pl.Boolean:
            return ["False", "True"]  # Same order as pandas.bool -> str
        if s.dtype == pl.Categorical and hasattr(s.cat, "get_categories"):
            cats = s.cat.get_categories().to_list()
        else:
            # Non-categorical column: get unique strings and sort naturally
            cats = s.cast(pl.Utf8).unique().to_list()
            cats = natsorted([str(x) for x in cats])
        return [str(x) for x in cats]

    # Fallback: any array-like
    arr = np.asarray(s, dtype=object)
    arr = arr[~pd.isnull(arr)]
    return [str(x) for x in natsorted(np.unique(arr))]

def _set_colors_for_categorical_obs(
    adata, value_to_plot: str, palette: Union[str, Sequence[str], Cycler, Mapping[str, str]]
):
    """Set `adata.uns[f'{value_to_plot}_colors']` according to the given palette.
    Compatible with pandas/Polars; if palette is dict, it will match by category keys.
    """
    cats = _obs_categories_ordered(adata, value_to_plot)
    n = len(cats)
    color_key = f"{value_to_plot}_colors"

    # 1) Handle different types of palette, generate color list of length n
    if isinstance(palette, Mapping):
        # dict: {category: color}
        # Missing categories use default color
        base_cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color", None) or [
            to_hex(mpl.colormaps["tab20"](i / 19)) for i in range(20)
        ]
        colors_list = []
        for i, cat in enumerate(cats):
            c = palette.get(cat, base_cycle[i % len(base_cycle)])
            colors_list.append(c)

    elif isinstance(palette, str) and (palette in mpl.colormaps):
        cmap = mpl.colormaps[palette]
        denom = max(n - 1, 1)
        colors_list = [to_hex(cmap(i / denom), keep_alpha=True) for i in range(n)]

    else:
        # Sequence or Cycler
        if isinstance(palette, Sequence) and not isinstance(palette, str):
            # Validate colors and convert to Cycler
            try:
                _color_list = [
                    (color if is_color_like(color) else additional_colors[color])
                    for color in palette
                ]
            except KeyError as e:
                raise ValueError(
                    f"The following color value of the given palette is not valid: {e.args[0]!r}"
                ) from None
            if len(_color_list) < n:
                logg.warning(
                    "Length of palette colors is smaller than the number of categories "
                    f"(palette length: {len(_color_list)}, categories length: {n}). "
                    "Some categories will have the same color."
                )
            from cycler import Cycler, cycler
            palette = cycler(color=_color_list)

        if not isinstance(palette, Cycler):
            raise ValueError(
                "Please check that 'palette' is a valid matplotlib colormap name, "
                "a list/tuple of colors, or a cycler with key='color'."
            )
        if "color" not in palette.keys:
            raise ValueError("Please set the palette key 'color'.")

        cc = palette()
        colors_list = [to_hex(next(cc)["color"], keep_alpha=True) for _ in range(n)]

    # 2) Convert to hex and write to adata.uns
    _uns_put_colors_dual(adata, color_key, [to_hex(c, keep_alpha=True) for c in colors_list])


def _uns_supports_str_array(uns) -> bool:
    """Check if .uns can safely read back string arrays. Rust backends typically return False."""
    try:
        uns["_ov_probe"] = np.asarray(["#000000"], dtype="U9")
        _ = uns["_ov_probe"]  # Read it back
        del uns["_ov_probe"]
        return True
    except Exception:
        try:
            del uns["_ov_probe"]
        except Exception:
            pass
        return False


def _uns_put_colors_dual(adata, name: str, colors_list):
    """
    Dual-track writing:
    - Always write: {name}_colors_rgba -> (n,4) float32 RGBA (Rust backend stable)
    - If possible: {name}_colors -> <U... string array (Python anndata friendly)
    """
    # Convert to hex
    hex_list = [
        c if (isinstance(c, str) and c.startswith("#")) else to_hex(c, keep_alpha=True)
        for c in colors_list
    ]
    # Always write RGBA (Rust/pyanndata most stable)
    rgba = np.asarray([to_rgba(h) for h in hex_list], dtype=np.float32)
    adata.uns[f"{name}_colors_rgba"] = rgba

    # If string array reading is supported, also write ..._colors
    adata.uns[f"{name}_colors"] = np.asarray(hex_list, dtype="U16")


def _uns_read_colors_dual(adata, name: str):
    """
    Read colors: try ..._colors (string) first, fall back to ..._colors_rgba (float32) and convert back to hex.
    Returns List[str] (#RRGGBB(AA))
    """
    # Try string array first (Python anndata case)
    try:
        v = adata.uns[f"{name}_colors"]
        if hasattr(v, "to_list"):  # Polars Series
            v = v.to_list()
        arr = np.asarray(v)
        if arr.dtype.kind in ("U", "S", "O"):
            return [str(x) for x in (arr.tolist() if isinstance(arr, np.ndarray) else list(arr))]
    except Exception:
        pass  # Rust backend may throw PanicException or TypeError

    # Fall back to RGBA
    v = adata.uns[f"{name}_colors_rgba"]
    arr = v.to_numpy() if hasattr(v, "to_numpy") else np.asarray(v)
    return [to_hex(tuple(row), keep_alpha=True) for row in arr]
