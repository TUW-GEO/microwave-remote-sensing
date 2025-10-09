"""Functions for plotting."""

from functools import partial

import holoviews as hv
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import rioxarray  # noqa: F401
from holoviews.streams import RangeXY

hv.extension("bokeh")


def handles_(color_mapping, present_landcover_codes):
    return [
        mpatches.Patch(
            color=info["color"], label=(f"{info['value']} - " + (info["label"]))
        )
        for info in color_mapping.values()
        if info["value"] in present_landcover_codes
    ]


def plot_corine_data(cor_da, cmap, norm, color_mapping, present_landcover_codes):
    cor_da.plot(
        figsize=(10, 10), cmap=cmap, norm=norm, add_colorbar=False
    ).axes.set_aspect("equal")

    handles = handles_(color_mapping, present_landcover_codes)
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
    """Use for Callback Function Landcover.

    Parameters
    ----------
    time: panda.datetime
        time slice
    land_cover: int
        land cover type
    x_range: array_like
        longitude range
    y_range: array_like
        latitude range
    var_ds: xarray.Dataset
        input data. Defaults to None.

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

    return hv.opts.Image(
        cmap="Greys_r",
        colorbar=True,
        tools=["hover"],
        clim=(robust_min, robust_max),
        aspect="equal",
        framewise=False,
        frame_height=500,
        frame_width=500,
    )


hist_opts = hv.opts.Histogram(width=350, height=555)


def plot_variability_over_time(color_mapping, var_ds, present_landcover_codes):
    robust_min = var_ds.sig0.quantile(0.02).item()
    robust_max = var_ds.sig0.quantile(0.98).item()

    bin_edges = bin_edges_(robust_min, robust_max)

    land_cover.update(
        {
            f"{int(value): 02} {color_mapping[value]['label']}": int(value)
            for value in present_landcover_codes
        }
    )
    time = var_ds.sig0["time"].values

    load_image_partial = partial(load_image, var_ds=var_ds)

    dmap = (
        hv.DynamicMap(
            load_image_partial, kdims=["Time", "Landcover"], streams=[rangexy]
        )
        .redim.values(Time=time, Landcover=land_cover)
        .hist(normed=True, bins=bin_edges)
    )

    image_opts = image_opts_(var_ds)

    return dmap.opts(image_opts, hist_opts)
