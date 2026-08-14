"""Grouped statistical plots: bar+dot, box, violin."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
from anndata import AnnData
from matplotlib.axes import Axes
from scipy.stats import kruskal

def bardotplot(
    adata,
    groupby,
    color,
    figsize=(8, 3),
    return_values=False,
    fontsize=12,
    xlabel="",
    ylabel="",
    xticks_rotation=90,
    ax=None,
    bar_kwargs=None,
    scatter_kwargs=None,
):
    if bar_kwargs is None:
        bar_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}

    var_ticks = False
    obs_ticks = False
    plot_text_ = color
    if plot_text_ in adata.var_names:
        adata1 = adata
        var_ticks = True
    elif plot_text_ in adata.obs.columns:
        adata1 = adata
        obs_ticks = True
    elif (adata.raw != None) and (plot_text_ in adata.raw.var_names):
        adata1 = adata1.raw.to_adata()
        var_ticks = True
    else:
        print(f"Please check the `{color}` key in adata.obs or adata.var")
        return
    adata1.obs[groupby] = adata1.obs[groupby].astype("category")

    if var_ticks == True:
        plot_data = pd.DataFrame()
        max_len = 0
        for group in adata1.obs[groupby].cat.categories:
            if max_len < len(adata1[adata1.obs[groupby] == group, plot_text_].to_df().values.reshape(-1)):
                max_len = len(adata1[adata1.obs[groupby] == group, plot_text_].to_df().values.reshape(-1))
        for group in adata1.obs[groupby].cat.categories:
            t_data1 = list(adata1[adata1.obs[groupby] == group, plot_text_].to_df().values.reshape(-1))
            while len(t_data1) < max_len:
                t_data1.append(np.nan)
            plot_data[group] = t_data1
    elif obs_ticks == True:
        plot_data = pd.DataFrame()
        max_len = 0
        for group in adata1.obs[groupby].cat.categories:
            if max_len < len(adata1.obs.loc[adata1.obs[groupby] == group, plot_text_].values.reshape(-1)):
                max_len = len(adata1.obs.loc[adata1.obs[groupby] == group, plot_text_].values.reshape(-1))
        for group in adata1.obs[groupby].cat.categories:
            t_data1 = list(adata1.obs.loc[adata1.obs[groupby] == group, plot_text_].values.reshape(-1))
            while len(t_data1) < max_len:
                t_data1.append(np.nan)
            plot_data[group] = t_data1

    if return_values == True:
        return plot_data

    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)

    xbar = np.arange(len(plot_data.columns.to_numpy()))
    # color_list_dot=[ov.utils.green_color[0],ov.utils.green_color[1],ov.utils.red_color[0],'#EC9DC5','#5BC23D']
    # color_list_dot=adata.uns['clusters_colors']
    if "{}_colors".format(groupby) in adata.uns.keys():
        color_list_dot = adata.uns["{}_colors".format(groupby)]
    else:
        if len(adata.obs[groupby].cat.categories) > 28:
            color_list_dot = sc.pl.palettes.default_102
        else:
            color_list_dot = sc.pl.palettes.zeileis_28

    plt.bar(
        x=plot_data.columns,
        height=plot_data.describe().loc["mean"],
        yerr=plot_data.sem(),
        color=color_list_dot,
        zorder=1,  # fill=False,
        edgecolor=color_list_dot,
        error_kw={"elinewidth": None, "capthick": None},
        **bar_kwargs,
    )
    bw = 0.4
    for cols in range(len(plot_data.columns.to_numpy())):
        # get markers from here https://matplotlib.org/3.1.1/api/markers_api.html
        plt.scatter(
            x=np.linspace(xbar[cols] - bw / 2, xbar[cols] + bw / 2, int(plot_data.describe().loc["count"][cols])),
            y=plot_data[plot_data.columns[cols]].dropna(),
            color=color_list_dot[cols],
            zorder=1,
            **scatter_kwargs,
        )

    plt.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    plt.xticks(rotation=xticks_rotation, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize + 1)
    plt.ylabel(ylabel, fontsize=fontsize + 1)
    plt.title(plot_text_, fontsize=fontsize + 1)
    if ax == None:
        return fig, ax




def single_group_boxplot(
    adata,
    groupby: str = "",
    color: str = "",
    type_color_dict: dict = None,
    title: str = "",
    ylabel: str = "",
    kruskal_test: bool = False,
    figsize: tuple = (4, 4),
    x_ticks_plot: bool = False,
    legend_plot: bool = True,
    bbox_to_anchor: tuple = (1, 0.55),
    save: bool = False,
    point_number: int = 5,
    save_pathway: str = "",
    sort: bool = True,
    scatter_kwargs: dict = None,
    ax=None,
    fontsize=12,
):
    """
    adata (AnnData object): The data object containing the information for plotting.
    groupby (str): The variable used for grouping the data.
    color (str): The variable used for coloring the data points.
    type_color_dict (dict): A dictionary mapping group categories to specific colors.
    title (str): The title for the plot.
    ylabel (str): The label for the y-axis.
    kruskal_test (bool): Whether to perform a Kruskal-Wallis test and display the p-value on the plot.
    figsize (tuple): The size of the plot figure in inches (width, height).
    x_ticks_plot (bool): Whether to display x-axis tick labels.
    legend_plot (bool): Whether to display a legend for the groups.
    bbox_to_anchor (tuple): The position of the legend bbox (x, y) in axes coordinates.
    save (bool): Whether to save the plot to a file.
    point_number (int): The number of data points to be plotted for each group.
    save_pathway (str): The file path for saving the plot (if save is True).
    sort (bool): Whether to sort the groups based on their mean values.
    scatter_kwargs (dict): Additional keyword arguments for customizing the scatter plot.
    ax (matplotlib.axes.Axes): A pre-existing axes object for plotting (optional).

    Example:
    ov.pl.single_group_boxplot(adata,groupby='clusters',
             color='Sox_aucell',
             type_color_dict=dict(zip(pd.Categorical(adata.obs['clusters']).categories, adata.uns['clusters_colors'])),
             x_ticks_plot=True,
             figsize=(5,4),
             kruskal_test=True,
             ylabel='Sox_aucell',
             legend_plot=False,
             bbox_to_anchor=(1,1),
             title='Expression',
             scatter_kwargs={'alpha':0.8,'s':10,'marker':'o'},
             point_number=15,
             sort=False,
             save=False,
             )
    """

    if scatter_kwargs is None:
        scatter_kwargs = {}

    # Create an empty dictionary to store results
    plot_data = {}

    var_ticks = False
    obs_ticks = False
    plot_text_ = color
    if plot_text_ in adata.var_names:
        adata1 = adata.copy()
        var_ticks = True
    elif plot_text_ in adata.obs.columns:
        adata1 = adata.copy()
        obs_ticks = True
    elif (adata.raw is not None) and (plot_text_ in adata.raw.var_names):
        adata1 = adata.raw.to_adata().copy()
        var_ticks = True
    else:
        print(f"Please check the `{color}` key in adata.obs or adata.var")
        return
    adata1.obs[groupby] = adata1.obs[groupby].astype("category")

    if var_ticks == True:
        adata1.obs[color] = adata1[:, plot_text_].to_df().values.flatten()
        # print(adata1.obs[color])

    # Categorize by groups

    for group in adata1.obs[groupby].cat.categories:
        plot_data[group] = np.array(adata1.obs.loc[adata1.obs[groupby] == group, color].tolist())

    if sort == True:
        sorted_keys = sorted(plot_data.keys(), key=lambda k: np.mean(plot_data[k]))
        sorted_plot_data = {key: plot_data[key] for key in sorted_keys}
        plot_data = sorted_plot_data

        sorted_colors = [type_color_dict[key] for key in sorted_keys]
        sc_color = sorted_colors
    else:
        sc_color = [type_color_dict[key] for key in plot_data.keys()]

    shake_dict = {}

    for group in adata1.obs[groupby].cat.categories:
        data_list = []
        gene_data = adata1.obs.loc[adata1.obs[groupby] == group, color].tolist()
        if len(gene_data) > point_number:
            bootstrap_data = np.random.choice(gene_data, size=point_number, replace=False)
        else:
            bootstrap_data = gene_data
        shake_dict[group] = np.array(bootstrap_data)

    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot boxplots
    width = 0.8
    ticks = np.arange(len(plot_data))
    positions = np.arange(len(ticks))

    for num, (hue_data, hue_color) in enumerate(zip(plot_data.keys(), sc_color)):
        position = positions[num]
        b1 = ax.boxplot(plot_data[hue_data], positions=[position], sym="", widths=width, patch_artist=True)
        plt.scatter(
            np.random.normal(position, 0.12, point_number),
            shake_dict[hue_data],
            c=hue_color,
            zorder=1,
            **scatter_kwargs,
        )
        box = b1["boxes"][0]
        light_hue_color = tuple((min(1, c + 0.5 * (1 - c))) for c in plt.cm.colors.to_rgb(hue_color))
        box.set(facecolor=light_hue_color, edgecolor=hue_color, linewidth=2)
        plt.setp(b1["whiskers"], color=hue_color, linewidth=2)
        plt.setp(b1["caps"], color=hue_color, linewidth=2)
        plt.setp(b1["medians"], color=hue_color, linewidth=3)

    # Axis labels and title

    if x_ticks_plot == True:
        ax.set_xticks(positions)
        ax.set_xticklabels(plot_data.keys(), rotation=90, fontsize=fontsize)
    else:
        ax.set_xticklabels([])

    yticks = ax.get_yticks()
    ax.set_title(
        title,
        fontsize=fontsize + 1,
    )
    plt.ylabel(
        ylabel,
        fontsize=fontsize + 1,
    )
    plt.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    if legend_plot == True:
        labels = list(plot_data.keys())
        patches = [mpatches.Patch(color=sc_color[i], label="{:s}".format(labels[i])) for i in range(len(plot_data))]
        ax.legend(handles=patches, bbox_to_anchor=bbox_to_anchor, ncol=1, fontsize=fontsize)

    if kruskal_test == True:
        data_list = [plot_data[key] for key in plot_data]
        statistic, p_value = kruskal(*data_list)

        if p_value < 0.0001:
            formatted_p_value = "{:.2e}".format(p_value)
        else:
            formatted_p_value = "{:.4f}".format(p_value)
        if p_value < 2.2e-16:
            formatted_p_value = 2.2e-16
            text = f"Kruskal-Wallis: P < {formatted_p_value}"
        else:
            text = f"Kruskal-Wallis: P = {formatted_p_value}"
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0.5"),
        )

    if save == True:
        plt.savefig(save_pathway, dpi=300, bbox_inches="tight")

    if ax is None:
        return fig, ax




def plot_boxplots(  # pragma: no cover
    data,
    feature_name: str,
    modality_key: str = "coda",
    y_scale="relative",
    plot_facets: bool = False,
    add_dots: bool = False,
    cell_types=None,
    args_boxplot=None,
    args_swarmplot=None,
    palette="Blues",
    show_legend=True,
    level_order=None,
    figsize=None,
    dpi=100,
    return_fig=None,
    ax=None,
    show=None,
    save=None,
):
    """Grouped boxplot visualization.

         The cell counts for each cell type are shown as a group of boxplots
         with intra--group separation by a covariate from data.obs.

        Args:
            data: AnnData object or MuData object
            feature_name: The name of the feature in data.obs to plot
            modality_key: If data is a MuData object, specify which modality to use.
            y_scale: Transformation to of cell counts. Options: "relative" - Relative abundance, "log" - log(count),
                     "log10" - log10(count), "count" - absolute abundance (cell counts).
            plot_facets: If False, plot cell types on the x-axis. If True, plot as facets.
            add_dots: If True, overlay a scatterplot with one dot for each data point.
            cell_types: Subset of cell types that should be plotted.
            args_boxplot: Arguments passed to sns.boxplot.
            args_swarmplot: Arguments passed to sns.swarmplot.
            figsize: Figure size.
            dpi: Dpi setting.
            palette: The seaborn color map for the barplot.
            show_legend: If True, adds a legend.
            level_order: Custom ordering of bars on the x-axis.

        Returns:
            Depending on `plot_facets`, returns a :class:`~matplotlib.axes.Axes` (`plot_facets = False`)
            or :class:`~sns.axisgrid.FacetGrid` (`plot_facets = True`) object

        Examples:
            >>> import pertpy as pt
            >>> haber_cells = pt.dt.haber_2017_regions()
            >>> sccoda = pt.tl.Sccoda()
            >>> mdata = sccoda.load(haber_cells, type="cell_level", generate_sample_level=True, cell_type_identifier="cell_label", \
                sample_identifier="batch", covariate_obs=["condition"])
            >>> sccoda.plot_boxplots(mdata, feature_name="condition", add_dots=True)

        Preview:
            .. image:: /_static/docstring_previews/sccoda_boxplots.png
        """
    if args_boxplot is None:
        args_boxplot = {}
    if args_swarmplot is None:
        args_swarmplot = {}
    # if isinstance(data, MuData):
    #    data = data[modality_key]
    # if isinstance(data, AnnData):
    #    data = data
    # y scale transformations
    if y_scale == "relative":
        sample_sums = np.sum(data.X, axis=1, keepdims=True)
        X = data.X / sample_sums
        value_name = "Proportion"
    # add pseudocount 0.5 if using log scale
    elif y_scale == "log":
        X = data.X.copy()
        X[X == 0] = 0.5
        X = np.log(X)
        value_name = "log(count)"
    elif y_scale == "log10":
        X = data.X.copy()
        X[X == 0] = 0.5
        X = np.log(X)
        value_name = "log10(count)"
    elif y_scale == "count":
        X = data.X
        value_name = "count"
    else:
        raise ValueError("Invalid y_scale transformation")

    count_df = pd.DataFrame(X, columns=data.var.index, index=data.obs.index).merge(
        data.obs[feature_name], left_index=True, right_index=True
    )
    plot_df = pd.melt(count_df, id_vars=feature_name, var_name="Cell type", value_name=value_name)
    if cell_types is not None:
        plot_df = plot_df[plot_df["Cell type"].isin(cell_types)]

    # Currently disabled because the latest statsannotations does not support the latest seaborn.
    # We had to drop the dependency.
    # Get credible effects results from model
    # if draw_effects:
    #     if model is not None:
    #         credible_effects_df = model.credible_effects(data, modality_key).to_frame().reset_index()
    #     else:
    #         print("[bold yellow]Specify a tasCODA model to draw effects")
    #     credible_effects_df[feature_name] = credible_effects_df["Covariate"].str.removeprefix(f"{feature_name}[T.")
    #     credible_effects_df[feature_name] = credible_effects_df[feature_name].str.removesuffix("]")
    #     credible_effects_df = credible_effects_df[credible_effects_df["Final Parameter"]]

    # If plot as facets, create a FacetGrid and map boxplot to it.
    if plot_facets:
        if level_order is None:
            level_order = pd.unique(plot_df[feature_name])

        K = X.shape[1]

        if figsize is not None:
            height = figsize[0]
            aspect = np.round(figsize[1] / figsize[0], 2)
        else:
            height = 3
            aspect = 2

        g = sns.FacetGrid(
            plot_df,
            col="Cell type",
            sharey=False,
            col_wrap=int(np.floor(np.sqrt(K))),
            height=height,
            aspect=aspect,
        )
        g.map(
            sns.boxplot,
            feature_name,
            value_name,
            palette=palette,
            order=level_order,
            **args_boxplot,
        )

        if add_dots:
            if "hue" in args_swarmplot:
                hue = args_swarmplot.pop("hue")
            else:
                hue = None

            if hue is None:
                g.map(
                    sns.swarmplot,
                    feature_name,
                    value_name,
                    color="black",
                    order=level_order,
                    **args_swarmplot,
                ).set_titles("{col_name}")
            else:
                g.map(
                    sns.swarmplot,
                    feature_name,
                    value_name,
                    hue,
                    order=level_order,
                    **args_swarmplot,
                ).set_titles("{col_name}")

        if save:
            plt.savefig(save, bbox_inches="tight")
        if show:
            plt.show()
        if return_fig:
            return plt.gcf()
        if not (show or save):
            return g
        return None

    # If not plot as facets, call boxplot to plot cell types on the x-axis.
    else:
        if level_order:
            args_boxplot["hue_order"] = level_order
            args_swarmplot["hue_order"] = level_order

        _, ax = plt.subplots(figsize=figsize, dpi=dpi)

        ax = sns.boxplot(
            x="Cell type",
            y=value_name,
            hue=feature_name,
            data=plot_df,
            fliersize=1,
            palette=palette,
            ax=ax,
            **args_boxplot,
        )

        # Currently disabled because the latest statsannotations does not support the latest seaborn.
        # We had to drop the dependency.
        # if draw_effects:
        #     pairs = [
        #         [(row["Cell Type"], row[feature_name]), (row["Cell Type"], "Control")]
        #         for _, row in credible_effects_df.iterrows()
        #     ]
        #     annot = Annotator(ax, pairs, data=plot_df, x="Cell type", y=value_name, hue=feature_name)
        #     annot.configure(test=None, loc="outside", color="red", line_height=0, verbose=False)
        #     annot.set_custom_annotations([row[feature_name] for _, row in credible_effects_df.iterrows()])
        #     annot.annotate()

        if add_dots:
            sns.swarmplot(
                x="Cell type",
                y=value_name,
                data=plot_df,
                hue=feature_name,
                ax=ax,
                dodge=True,
                palette="dark:black",
                **args_swarmplot,
            )

        cell_types = pd.unique(plot_df["Cell type"])
        ax.set_xticklabels(cell_types, rotation=90)

        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            handout = []
            labelout = []
            for h, l in zip(handles, labels, strict=False):
                if l not in labelout:
                    labelout.append(l)
                    handout.append(h)
            ax.legend(
                handout,
                labelout,
                loc="upper left",
                bbox_to_anchor=(1, 1),
                ncol=1,
                title=feature_name,
            )

        if save:
            plt.savefig(save, bbox_inches="tight")
        if show:
            plt.show()
        if return_fig:
            return plt.gcf()
        if not (show or save):
            return ax
        return None



def violin_old(
    adata, keys=None, groupby=None, ax=None, figsize=(4, 4), fontsize=13, ticks_fontsize=None, rotation=90, **kwargs
):
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)
    sc.pl.violin(adata, keys=keys, groupby=groupby, ax=ax, show=False, **kwargs)
    plt.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    if ticks_fontsize == None:
        ticks_fontsize = fontsize - 1

    plt.xticks(fontsize=ticks_fontsize, rotation=rotation)
    plt.yticks(fontsize=ticks_fontsize)
    plt.xlabel(groupby, fontsize=fontsize)
    plt.ylabel(keys, fontsize=fontsize)

    if ax == None:
        return fig, ax

    # plt.xticks(fontsize=ticks_fontsize,rotation=90)
    # plt.yticks(fontsize=ticks_fontsize)


def violin_box(adata, keys, groupby, ax=None, figsize=(4, 4), show=True, max_strip_points=1000):
    import colorcet
    from scipy.sparse import issparse

    # 获取 y 数据
    y = None
    if not adata.raw is None and keys in adata.raw.var_names:
        y = adata.raw[:, keys].X
    elif keys in adata.obs.columns:
        y = adata.obs[keys].values
    elif keys in adata.var_names:
        y = adata[:, keys].X
    else:
        raise ValueError(f"{keys} not found in adata.raw.var_names, adata.var_names, or adata.obs.columns")

    if issparse(y):
        y = y.toarray().reshape(-1)
    else:
        y = y.reshape(-1)

    # 获取 x 数据
    x = adata.obs[groupby].values.reshape(-1)

    # 创建绘图数据
    plot_data = pd.DataFrame({groupby: x, keys: y})

    # 创建图形和轴
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 获取或设置颜色
    from ._palettes import palette_28, palette_56, palette_112, sc_color

    if f"{groupby}_colors" not in adata.uns or adata.uns[f"{groupby}_colors"] is None:
        # colors = ['#%02x%02x%02x' % tuple([int(k * 255) for k in i]) for i in colorcet.glasbey_bw_minc_20_maxl_70]
        if len(adata.obs[groupby].unique()) > 56:
            colors = palette_112
        elif len(adata.obs[groupby].unique()) > 28:
            colors = palette_56
        else:
            colors = sc_color
        adata.uns[f"{groupby}_colors"] = colors[: len(adata.obs[groupby].unique())]

    # 绘制小提琴图
    sns.violinplot(
        x=groupby,
        y=keys,
        data=plot_data,
        hue=groupby,
        dodge=False,
        palette=adata.uns[f"{groupby}_colors"],
        scale="width",
        inner=None,
        ax=ax,
        legend=False,
    )

    # 调整小提琴图
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))

    # 限制 stripplot 的数据点数量
    if len(plot_data) > max_strip_points:
        plot_data = plot_data.sample(max_strip_points)

    # 绘制 stripplot
    old_len_collections = len(ax.collections)
    sns.stripplot(
        x=groupby,
        y=keys,
        data=plot_data,
        hue=groupby,
        palette=adata.uns[f"{groupby}_colors"],
        dodge=False,
        ax=ax,
    )

    # 调整 stripplot 点的位置
    for dots in ax.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([0.12, 0]))

    # 绘制箱线图
    sns.boxplot(
        x=groupby,
        y=keys,
        data=plot_data,
        saturation=1,
        showfliers=False,
        width=0.3,
        boxprops={"zorder": 3, "facecolor": "none"},
        ax=ax,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    # remove legend
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    # ax.legend().set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    #

    if show:
        plt.show()

    return ax



def dotplot_doublegroup(adata, gene, group1, group2, cmap="Reds", standard_scale="group", figsize=(6, 4), layer=None):
    # 检查输入
    if gene not in adata.var_names:
        raise ValueError(f"Gene '{gene}' is not in the provided AnnData object.")

    if group1 not in adata.obs.columns:
        raise ValueError(f"Group '{group1}' is not in the provided AnnData object.")

    if group2 not in adata.obs.columns:
        raise ValueError(f"Group '{group2}' is not in the provided AnnData object.")

    # 将分组列转换为类别类型
    adata.obs[group1] = adata.obs[group1].astype("category")
    adata.obs[group2] = adata.obs[group2].astype("category")
    group1s = adata.obs[group1].cat.categories
    group2s = adata.obs[group2].cat.categories

    group_li_exp_mean_pd = pd.DataFrame(np.zeros((len(group1s), len(group2s))), index=group1s, columns=group2s)
    group_li_exp_size_pd = pd.DataFrame(np.zeros((len(group1s), len(group2s))), index=group1s, columns=group2s)

    for group1_ in group1s:
        adata1 = adata[adata.obs[group1] == group1_, [gene]]
        for group2_ in group2s:
            if layer is None:
                exp = adata1[adata1.obs[group2] == group2_, [gene]].to_df().values.reshape(-1)
            elif layer in adata1.layers.keys():
                exp = adata1[adata1.obs[group2] == group2_, [gene]].layers[layer].to_df().values.reshape(-1)
            else:
                raise ValueError(f"Layer '{layer}' is not in the provided AnnData object.")
            exp_larger_zero = exp[exp > 0]
            if len(exp) != 0:
                group_li_exp_size_pd.loc[group1_, group2_] = len(exp_larger_zero) / len(exp)
                group_li_exp_mean_pd.loc[group1_, group2_] = np.mean(exp)
            else:
                group_li_exp_size_pd.loc[group1_, group2_] = 0

    dot_color_df = group_li_exp_mean_pd

    if standard_scale == "group":
        dot_color_df = dot_color_df.sub(dot_color_df.min(1), axis=0)
        dot_color_df = dot_color_df.div(dot_color_df.max(1), axis=0).fillna(0)
    elif standard_scale == "var":
        dot_color_df -= dot_color_df.min(0)
        dot_color_df = (dot_color_df / dot_color_df.max(0)).fillna(0)
    else:
        pass

    # 设置常量
    dot_min = 0
    dot_max = 1
    size_exponent = 1
    largest_dot = 100
    smallest_dot = 10
    dot_edge_lw = 1
    size_title = "Fraction of cells\nin group (%)"

    # 创建图形和网格布局
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax = fig.add_subplot(gs[0])

    # 绘制圆点
    for y, (label, row_mean) in enumerate(dot_color_df.iterrows()):
        for x, (column, value_mean) in enumerate(row_mean.items()):
            value_size = group_li_exp_size_pd.loc[label, column]
            size = value_size * 500  # 调整大小
            color = plt.get_cmap(cmap)(value_mean / dot_color_df.values.max())
            ax.scatter(x, y, s=size, color=color, alpha=1, edgecolors="w", linewidth=0.5)

    # 设置轴
    ax.set_xticks(range(len(group_li_exp_mean_pd.columns)))
    ax.set_xticklabels(group_li_exp_mean_pd.columns, rotation=45)
    ax.set_yticks(range(len(group_li_exp_mean_pd.index)))
    ax.set_yticklabels(group_li_exp_mean_pd.index)
    ax.set_title("Dot Plot")

    # 添加颜色图例
    ax1 = fig.add_subplot(gs[1])
    legend_gs = ax1.get_subplotspec().subgridspec(10, 1)

    color_legend_ax = fig.add_subplot(legend_gs[7])
    mappable = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=group_li_exp_mean_pd.values.min(), vmax=group_li_exp_mean_pd.values.max())
    )
    plt.colorbar(mappable, cax=color_legend_ax, orientation="horizontal")
    color_legend_ax.set_title("Mean expression\nin group", fontsize="small")
    color_legend_ax.xaxis.set_tick_params(labelsize="small")

    # 添加点大小图例
    size_legend_ax = fig.add_subplot(legend_gs[2:5])
    diff = dot_max - dot_min
    step = 0.1 if 0.3 < diff <= 0.6 else 0.05 if diff <= 0.3 else 0.2

    size_range = np.arange(dot_max, dot_min, step * -1)[::-1]
    if dot_min != 0 or dot_max != 1:
        dot_range = dot_max - dot_min
        size_values = (size_range - dot_min) / dot_range
    else:
        size_values = size_range

    size = size_values**size_exponent
    size = size * (largest_dot - smallest_dot) + smallest_dot

    size_legend_ax.scatter(
        np.arange(len(size)) + 0.5,
        np.repeat(0, len(size)),
        s=size * 5,
        color="gray",
        edgecolor="black",
        linewidth=dot_edge_lw,
        zorder=100,
    )
    size_legend_ax.set_xticks(np.arange(len(size)) + 0.5)
    labels = [f"{np.round((x * 100), decimals=0).astype(int)}" for x in size_range]
    size_legend_ax.set_xticklabels(labels, fontsize="small")

    size_legend_ax.tick_params(axis="y", left=False, labelleft=False, labelright=False)

    size_legend_ax.spines["right"].set_visible(False)
    size_legend_ax.spines["top"].set_visible(False)
    size_legend_ax.spines["left"].set_visible(False)
    size_legend_ax.spines["bottom"].set_visible(False)
    size_legend_ax.grid(visible=False)

    ymax = size_legend_ax.get_ylim()[1]
    size_legend_ax.set_ylim(-1.05 - largest_dot * 0.003, 4)
    size_legend_ax.set_title(size_title, y=ymax + 0.45, size="small")

    xmin, xmax = size_legend_ax.get_xlim()
    size_legend_ax.set_xlim(xmin - 0.15, xmax + 0.5)

    ax.grid(False)
    ax1.grid(False)
    ax1.axis(False)
    plt.show()


