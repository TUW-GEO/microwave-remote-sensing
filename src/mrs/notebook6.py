import json

import holoviews as hv
import intake
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from holoviews.streams import RangeXY
from matplotlib.colors import BoundaryNorm, ListedColormap

hv.extension("bokeh")

def handles_(color_mapping, present_landcover_codes):
    return [
    mpatches.Patch(color=info["color"], label=(f"{info['value']} - " + (info["label"])))
    for info in color_mapping.values()
    if info["value"] in present_landcover_codes
]

def plot_cor_da(cor_da, cmap, norm, handles):
    cor_da.plot(figsize=(10, 10), cmap=cmap, norm=norm, add_colorbar=False).axes.set_aspect(
        "equal"
    )

    plt.legend(
        handles=handles,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=7,
    )
    plt.title("CORINE Land Cover (EPSG:27704)")


land_cover = {"\xa0\xa0\xa0 Complete Land Cover": 1}

rangexy = RangeXY()

def bin_edges_(robust_min, robust_max):
    return [
    i + j * 0.5
    for i in range(int(robust_min) - 2, int(robust_max) + 2)
    for j in range(2)
]

def load_image(time, land_cover, x_range, y_range, var_ds=None):
    """Callback Function Landcover.

    Parameters
    ----------
    time: panda.datatime
        time slice
    landcover: int
        land cover type
    x_range: array_like
        longitude range
    y_range: array_like
        latitude range

    Returns
    -------
    holoviews.Image

    """
    if land_cover == "\xa0\xa0\xa0 Complete Land Cover":
        sig0_selected_ds = var_ds.sig0.sel(time=time)

    else:
        land_cover_value = int(land_cover.split()[0])
        mask_ds = var_ds.land_cover == land_cover_value
        sig0_selected_ds = var_ds.sig0.sel(time=time).where(mask_ds)

    hv_ds = hv.Dataset(sig0_selected_ds)
    img = hv_ds.to(hv.Image, ["x", "y"])

    if x_range and y_range:
        img = img.select(x=x_range, y=y_range)

    return hv.Image(img)





def image_opts_(var_ds):
    robust_min = var_ds.sig0.quantile(0.02).item()
    robust_max = var_ds.sig0.quantile(0.98).item()

    image_opts = hv.opts.Image(
        cmap="Greys_r",
        colorbar=True,
        tools=["hover"],
        clim=(robust_min, robust_max),
        aspect="equal",
        framewise=False,
        frame_height=500,
        frame_width=500,
    )

    return image_opts

hist_opts = hv.opts.Histogram(width=350, height=555)
