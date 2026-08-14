"""Color palettes and visualization utilities for single-cell analysis."""

from typing import Iterable, List, Optional, Sequence, Union
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap, is_color_like, to_hex
try:
    from rich import print as rprint
except ImportError:
    rprint = print


sc_color=[
 '#1F577B', '#A56BA7', '#E0A7C8', '#E069A6', '#941456', 
 '#FCBC10', '#EF7B77', '#279AD7','#F0EEF0',
 '#EAEFC5', '#7CBB5F','#368650','#A499CC','#5E4D9A',
 '#78C2ED','#866017', '#9F987F','#E0DFED',
 '#01A0A7', '#75C8CC', '#F0D7BC', '#D5B26C', '#D5DA48',
 '#B6B812', '#9DC3C3', '#A89C92', '#FEE00C', '#FEF2A1']

red_color=['#F0C3C3','#E07370','#CB3E35','#A22E2A','#5A1713',
           '#D3396D','#8B0000', '#A52A2A', '#CD5C5C', '#DC143C' ]

green_color=['#91C79D','#8FC155','#56AB56','#2D5C33','#BBCD91',
             '#6E944A','#A5C953','#3B4A25','#010000']

orange_color=['#EFBD49','#D48F3E','#AC8A3E','#7D7237','#745228',
              '#E1C085','#CEBC49','#EBE3A1','#6C6331','#8C9A48','#D7DE61']

blue_color=['#1F577B', '#279AD7', '#78C2ED', '#01A0A7', '#75C8CC', '#9DC3C3',
            '#3E8CB1', '#52B3AD', '#265B58', '#5860A7', '#312C6C', '#4CC9F0']

purple_color=['#823d86','#825b94','#bb98c6','#c69bc6','#a69ac9',
              '#c5a6cc','#caadc4','#d1c3d4']

#more beautiful colors
# 28-color palettes with distinct neighboring colors
palette_28 = sc_color[:28]
# 56-color palette with clear transitions

# 112-color palette with distinct transitions
cet_g_bw = [
 '#d60000', '#8c3bff', '#018700', '#00acc6', '#97ff00', '#ff7ed1', '#6b004f', '#ffa52f', '#00009c', '#857067',
 '#004942', '#4f2a00', '#00fdcf', '#bcb6ff', '#95b379', '#bf03b8', '#2466a1', '#280041', '#dbb3af', '#fdf490',
 '#4f445b', '#a37c00', '#ff7066', '#3f806e', '#82000c', '#a37bb3', '#344d00', '#9ae4ff', '#eb0077', '#2d000a',
 '#5d90ff', '#00c61f', '#5701aa', '#001d00', '#9a4600', '#959ea5', '#9a425b', '#001f31', '#c8c300', '#ffcfff',
 '#00bd9a', '#3615ff', '#2d2424', '#df57ff', '#bde6bf', '#7e4497', '#524f3b', '#d86600', '#647438', '#c17287',
 '#6e7489', '#809c03', '#bd8a64', '#623338', '#cacdda', '#6beb82', '#213f69', '#a17eff', '#fd03ca', '#75bcfd',
 '#d8c382', '#cda3cd', '#6d4f00', '#006974', '#469e5d', '#93c6bf', '#f9ff00', '#bf5444', '#00643b', '#5b4fa8',
 '#521f64', '#4f5eff', '#7e8e77', '#b808f9', '#8a91c3', '#b30034', '#87607e', '#9e0075', '#ffddc3', '#500800',
 '#1a0800', '#4b89b5', '#00dfdf', '#c8fff9', '#2f3415', '#ff2646', '#ff97aa', '#03001a', '#c860b1', '#c3a136',
 '#7c4f3a', '#f99e77', '#566464', '#d193ff', '#2d1f69', '#411a34', '#af9397', '#629e99', '#bcdd7b', '#ff5d93',
 '#0f2823', '#b8bdac', '#743b64', '#0f000c', '#7e6ebc', '#9e6b3b', '#ff4600', '#7e0087', '#ffcd3d', '#2f3b42',
 '#fda5ff', '#89013d', '#752b01', '#0a8995', '#050052', '#8ed631', '#52c372', '#465970', '#570121', '#a52101',
 '#90934b', '#00421d', '#8000d1', '#2f263f', '#bf3883', '#f4ffd4', '#00d3ff', '#6900f7', '#9cbad1', '#79d8aa',
 '#69565d', '#006905', '#36369c', '#018246', '#441d18', '#07a5ef', '#ff802f', '#a754b8', '#675982', '#72ffff',
 '#d88701', '#bad3ff', '#8e362f', '#a7a080', '#007ce2', '#8e7e8e', '#994487', '#00f034', '#aeaac8', '#a06062',
 '#4b3a77', '#6b8282', '#f0dde6', '#ffbad3', '#38a523', '#b3ffa8', '#0c1107', '#d6526e', '#959efd', '#7c7e00',
 '#759eb8', '#db877e', '#111318', '#d482d4', '#9e00bf', '#dbefff', '#8eaa9a', '#706442', '#493b3d', '#084d5e',
 '#9cb844', '#d8ddd4', '#caff6b', '#b364eb', '#465d33', '#009e7c', '#c14100', '#4fbcba', '#d88ab1', '#5b72b5',
 '#4b4101', '#95825d', '#49748a', '#ff72ff', '#82691c', '#dbcfff', '#7e6bfd', '#627560', '#ffc191', '#595d00',
 '#e408e6', '#b8b1b6', '#d32d41', '#314236', '#d8a362', '#5b8a33', '#2f1f00', '#97e6d6', '#2a6256', '#cd724d',
 '#5d3d28', '#0059d8', '#ac93d6', '#6b1d93', '#b3015d', '#410046', '#9cffcf', '#e4489c', '#e2e246', '#dbe2a5',
 '#002859', '#aa5b82', '#0000db', '#4b4d50', '#dabfd4', '#004d99', '#87649e', '#691d1c', '#8e52c4', '#b8dadf',
 '#ddb3fd', '#7b4854', '#4b7200', '#440077', '#b15e00', '#91d185', '#54334b', '#69af85', '#aa93af', '#e65442',
 '#8e8c89', '#70ac50', '#aa7c74', '#00343b', '#240f13', '#e6af00', '#79ccdb', '#18133a', '#9c5238', '#ba7b31',
 '#b6ca93', '#310800', '#a39505', '#00daba', '#74a0dd', '#623b72', '#ffda8e', '#77b800', '#3f2f1c', '#578759',
 '#2d0021', '#f4a1d4', '#da00aa', '#752849', '#bce400', '#c3c15d'
]

palette_112 = cet_g_bw[:112]
palette_56 = cet_g_bw[:56]

# ============================================================================
# Color Palette Functions

# ============================================================================

def get_palette(name: str) -> List[str]:
    """Get a predefined color palette by name.

    Parameters
    ----------
    name : str
        Name of the palette. Options are dynamically detected from the module.

    Returns
    -------
    List[str]
        List of hex color codes.

    Examples
    --------
    >>> get_palette('immune')
    ['#1f77b4', '#17becf', '#9467bd', ...]

    >>> get_palette('default')
    ['#1f77b4', '#ff7f0e', '#2ca02c', ...]
    """
    current_module = sys.modules[__name__]
    palette_map: dict[str, list] = {}

    for key, value in vars(current_module).items():
        if key.startswith("_"):
            continue
        if key.endswith("_PALETTE") and isinstance(value, dict):
            palette_map[key.lower().replace("_palette", "")] = list(value.values())
        elif key.endswith("_PALETTE") and isinstance(value, list):
            palette_map[key.lower().replace("_palette", "")] = value
        elif (
            isinstance(value, list)
            and value
            and isinstance(value[0], str)
            and value[0].startswith("#")
        ):
            palette_map[key.lower()] = value

    palette_map.setdefault("extended", palette_map.get("sc_color", sc_color))

    name_key = name.lower().replace("_palette", "")
    if name_key not in palette_map:
        raise ValueError(
            f"Unknown palette '{name}'. Available palettes: {list(palette_map.keys())}"
        )

    return palette_map[name_key]


def create_palette_from_types(
    cell_types: List[str],
    base_palette: str = "extended"
) -> dict:
    """Create a color mapping for a list of cell types.

    Parameters
    ----------
    cell_types : List[str]
        List of cell type names.
    base_palette : str, optional
        Base palette to use. Default: 'extended'.

    Returns
    -------
    dict
        Dictionary mapping cell types to colors.

    Examples
    --------
    >>> create_palette_from_types(['T cell', 'B cell', 'Monocyte'])
    {'T cell': '#440154', 'B cell': '#482878', 'Monocyte': '#3e4989'}
    """
    base_colors = get_palette(base_palette)
    n_types = len(cell_types)

    if n_types > len(base_colors):
        # Need to extend the palette
        from matplotlib.colors import to_hex
        from matplotlib.cm import get_cmap
        cmap = get_cmap("tab20")
        extra_colors = [to_hex(cmap(i)) for i in range(n_types - len(base_colors))]
        colors = base_colors + extra_colors
    else:
        colors = base_colors[:n_types]

    return {cell_type: colors[i] for i, cell_type in enumerate(cell_types)}


# ============================================================================
# Color Visualization Function
# ============================================================================

def show_color(
    color_input: Union[str, tuple, list, Colormap],
    title: Optional[str] = None,
    n_colors: int = 256
):
    """Intelligently visualizes a color input.

    - If the input is a single color, it displays a block of that color.
    - If the input is a list of colors (a palette), it displays the color blocks side-by-side.
    - If the input is a colormap (cmap), it displays its continuous color gradient.

    Parameters
    ----------
    color_input : str, tuple, list, or matplotlib.colors.Colormap
        Can be a single color like 'red', '#FF0000', or (1, 0, 0);
        a palette (list of colors) like ['red', 'blue', 'green'];
        or a colormap name/object like 'viridis' or plt.get_cmap('plasma').
    title : str, optional
        The title for the plot. If None, a default is generated.
    n_colors : int, optional
        The number of colors to use when rendering a cmap gradient. Defaults to 256.

    Examples
    --------
    >>> show_color('red')
    # Displays a single red block

    >>> show_color(['red', 'blue', 'green'])
    # Displays three color blocks side-by-side

    >>> show_color('viridis')
    # Displays the viridis colormap gradient

    >>> show_color(IMMUNE_PALETTE)
    # Displays all immune cell type colors
    """
    # --- Case 1 & 2: Handle a single color or a palette (list of colors) ---
    colors_to_show = []
    default_title = ""

    # is_color_like() checks if the input can be interpreted as a color
    if is_color_like(color_input):
        colors_to_show = [color_input]  # Put the single color into a list
        default_title = "Single Color"
    # Handle dictionary input (palettes)
    elif isinstance(color_input, dict):
        colors_to_show = list(color_input.values())
        default_title = f"Palette ({len(colors_to_show)} colors)"
    # isinstance() checks if the input is a list
    elif isinstance(color_input, list):
        colors_to_show = color_input  # It's already a list of colors
        default_title = f"Color Palette ({len(colors_to_show)} colors)"

    if colors_to_show:
        n = len(colors_to_show)
        # Adjust figure size to better accommodate the number of colors
        fig, ax = plt.subplots(figsize=(max(n, 2), 1))
        for i, color in enumerate(colors_to_show):
            # Use fill_between to draw each color block
            ax.fill_between([i, i + 1], 0, 1, color=color)
        ax.set_xlim(0, n)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(title if title else default_title, fontsize=16)
        plt.tight_layout()
        plt.show()
        return  # Done, so exit the function

    # --- Case 3: Handle a colormap (cmap) ---
    try:
        # Try to get a colormap from the input
        if isinstance(color_input, str):
            cmap_name = color_input
        elif isinstance(color_input, Colormap):
            cmap_name = color_input.name
        else:
            # If it's none of the above, consider the input invalid
            raise ValueError("Input type is not a list, color-like, or Colormap")

        cmap = plt.get_cmap(color_input)

        # Create an array from 0 to 1 to generate the gradient
        gradient = np.linspace(0, 1, n_colors)
        gradient = np.vstack((gradient, gradient))  # Reshape into a 2D array for imshow

        fig, ax = plt.subplots(figsize=(8, 1.5))
        # Use imshow to plot the gradient
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.axis('off')
        final_title = title if title else f"Colormap: '{cmap_name}'"
        ax.set_title(final_title, fontsize=16)
        plt.tight_layout()
        plt.show()
        return  # Done, so exit the function

    except (ValueError, TypeError) as e:
        rprint(
            f"Input '{color_input}' could not be identified as a valid color, "
            f"palette, or cmap. Error: {e}"
        )


# ============================================================================
# Palette Display Functions
# ============================================================================

def show_palette(
    palette: Union[str, List[str], dict],
    title: Optional[str] = None,
    labels: Optional[List[str]] = None
):
    """Display a palette with optional labels.

    Parameters
    ----------
    palette : str, List[str], or dict
        Palette name, list of colors, or dictionary mapping labels to colors.
    title : str, optional
        Title for the plot.
    labels : List[str], optional
        Labels for each color. Only used if palette is a list.

    Examples
    --------
    >>> show_palette('immune')
    # Displays immune cell palette

    >>> show_palette(IMMUNE_PALETTE)
    # Displays immune palette with cell type labels

    >>> show_palette(['red', 'blue', 'green'], labels=['A', 'B', 'C'])
    # Displays custom palette with labels
    """
    # Handle palette name
    if isinstance(palette, str):
        palette = get_palette(palette)
        if title is None:
            title = f"Palette: {palette}"

    # Handle dictionary input
    if isinstance(palette, dict):
        labels = list(palette.keys())
        colors = list(palette.values())
        if title is None:
            title = f"Palette ({len(palette)} colors)"
    else:
        colors = palette
        if labels is None:
            labels = [f"{i}" for i in range(len(colors))]
        if title is None:
            title = f"Palette ({len(colors)} colors)"

    n = len(colors)
    fig, ax = plt.subplots(figsize=(max(n * 0.5, 6), 2))

    # Draw color blocks
    for i, color in enumerate(colors):
        ax.fill_between([i, i + 1], 0, 1, color=color)
        # Add label
        if labels and i < len(labels):
            ax.text(i + 0.5, 0.5, labels[i], ha='center', va='center',
                   fontsize=9, rotation=45, color='white', weight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(title, fontsize=14, pad=10)
    plt.tight_layout()
    plt.show()


def list_available_palettes():
    """List all available predefined palettes dynamically.

    Returns
    -------
    dict
        Dictionary of available palettes and their descriptions.
    """
    # Dynamically detect all palette variables in the current module
    current_module = sys.modules[__name__]
    palettes = {
        key.lower().replace("_palette", ""): f"{key.replace('_PALETTE', '').replace('_', ' ').capitalize()} palette"
        for key, value in vars(current_module).items()
        if key.endswith("_PALETTE") and isinstance(value, (list, dict))
    }

    # print available palettes
    rprint("Available predefined palettes:")
    rprint("-" * 60)
    for name, desc in palettes.items():
        rprint(f"  {name:25s} : {desc}")
    rprint("-" * 60)
    rprint(f"\nTotal: {len(palettes)} palettes")

    return palettes


def reorder_and_set_palettes(adata, group, order=None, palette=None, group_color_dict=None):
    """
    Reorder a group column in adata.obs and set its corresponding palette.
    Supports both `order` and `palette` or a `group_color_dict`.

    Non-categorical columns are coerced to ``category`` automatically. When
    ``group_color_dict`` is used, only overlapping categories are applied
    (extras in the dict are skipped; categories missing from the dict get grey).

    Parameters
    ----------
    adata : AnnData
        The AnnData object to modify.
    group : str
        The name of the column in adata.obs to reorder.
    order : list, optional
        The new order of the categories. Ignored if `group_color_dict` is provided.
    palette : list, optional
        The list of colors corresponding to the new order. Ignored if `group_color_dict` is provided.
    group_color_dict : dict, optional
        A dictionary mapping group names to colors. If provided, `order` and `palette` are ignored.

    Returns
    -------
    None
        Modifies the adata object in place.
    """
    # Ensure the group exists in adata.obs
    if group not in adata.obs:
        raise ValueError(f"Group '{group}' not found in adata.obs.")

    # Coerce to categorical if needed (MuData mods often store labels as object/str)
    if not isinstance(adata.obs[group].dtype, pd.CategoricalDtype):
        adata.obs[group] = adata.obs[group].astype(str).astype("category")

    if isinstance(palette, dict) and group_color_dict is None:
        group_color_dict = palette
        palette = None

    # Handle group_color_dict
    if group_color_dict is not None:
        if not isinstance(group_color_dict, dict):
            raise ValueError("`group_color_dict` must be a dictionary.")
        existing = set(adata.obs[group].cat.categories.astype(str))
        # Keep dict key order, but only categories present in this object
        order = [k for k in group_color_dict.keys() if str(k) in existing]
        palette = [group_color_dict[k] for k in order]
        missing_in_data = [k for k in group_color_dict.keys() if str(k) not in existing]
        missing_in_dict = [c for c in adata.obs[group].cat.categories.astype(str) if c not in group_color_dict]
        if missing_in_data:
            print(
                f"Note: {len(missing_in_data)} palette key(s) not in '{group}' "
                f"(skipped): {missing_in_data}"
            )
        if missing_in_dict:
            # Append unmapped categories so they are not dropped from obs
            order = order + missing_in_dict
            palette = palette + ["#CCCCCC"] * len(missing_in_dict)
            print(
                f"Warning: {len(missing_in_dict)} categor(ies) in '{group}' "
                f"missing from palette (grey fallback): {missing_in_dict}"
            )
        if not order:
            raise ValueError(
                f"None of the keys in `group_color_dict` overlap with '{group}' categories: "
                f"{list(adata.obs[group].cat.categories)}"
            )
    else:
        if order is None or palette is None:
            raise ValueError("Provide either `group_color_dict`, or both `order` and `palette`.")
        existing_categories = set(adata.obs[group].cat.categories.astype(str))
        if set(map(str, order)) != existing_categories:
            raise ValueError(
                f"The provided order does not match the existing categories in '{group}'.\n"
                f"Existing categories: {list(adata.obs[group].cat.categories)}\n"
                f"Provided order: {order}"
            )

    # Ensure the palette length matches the order length
    if len(order) != len(palette):
        raise ValueError(
            f"The length of the palette ({len(palette)}) must match the length of the order ({len(order)})."
        )

    # Reorder the categories
    adata.obs[group] = adata.obs[group].astype(pd.CategoricalDtype(categories=order, ordered=True))

    # Set the palette in adata.uns
    adata.uns[f"{group}_colors"] = np.array(palette)

    print(f"Reordered '{group}' and set the palette successfully.")

def order_labels(observed=None, *, order=None):
    """Return ``order`` first, then any extra observed labels."""
    base = list(order or [])
    if observed is None:
        return base
    obs = [str(x) for x in observed]
    ordered = [c for c in base if c in obs]
    seen = set(ordered)
    for c in obs:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered
