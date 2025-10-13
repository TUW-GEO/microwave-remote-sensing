"""Functions for plotting."""

import base64
from functools import partial
from io import BytesIO

import folium
import holoviews as hv
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import rioxarray  # noqa: F401
import seaborn as sns
from holoviews.streams import RangeXY

hv.extension("bokeh")
land_cover = {"\xa0\xa0\xa0 Complete Land Cover": 1}
rangexy = RangeXY()
hist_opts = hv.opts.Histogram(width=350, height=555)
cmap_hls_hex = sns.color_palette("hls", n_colors=256).as_hex()
cmap_hls = sns.hls_palette(as_cmap=True)


def handles_(color_mapping, present_landcover_codes):
    """Create handles for plot."""
    return [
        mpatches.Patch(
            color=info["color"], label=(f"{info['value']} - " + (info["label"]))
        )
        for info in color_mapping.values()
        if info["value"] in present_landcover_codes
    ]


def plot_corine_data(cor_da, cmap, norm, color_mapping, present_landcover_codes):
    """Plot using CORINE Land Cover data."""
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


def bin_edges_(robust_min, robust_max):
    """Define the edges based of robust min and max values."""
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
    """Define robust min and max values and return image opts."""
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


def plot_variability_over_time(color_mapping, var_ds, present_landcover_codes):
    """Plot that shows variability over time."""
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


def plot_slc_all(datasets):
    """Plot Single Look Complex (SLC) data wih the 3 datasets: IW1, IW2, IW3."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 7), sharey=True)

    val_range = dict(vmin=0, vmax=255, cmap="gray")  # noqa C408

    for i, ds in enumerate(datasets):
        im = ds.intensity.plot(ax=ax[i], add_colorbar=False, **val_range)
        ax[i].tick_params(axis="both", which="major")

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.9, pad=0.2)  # noqa F841

    plt.show()


def plot_slc_iw2(iw2_ds):
    """Plot Single Look Complex (SLC) data wih only one datasets: IW2."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # noqa RUF059

    iw2_ds.intensity.plot(ax=axes[0], cmap="gray", robust=True)
    axes[0].set_title("Intensity Measurement of IW2")

    iw2_ds.phase.plot(ax=axes[1], cmap=cmap_hls)
    axes[1].set_title("Phase Measurement of IW2")

    plt.tight_layout()


def plot_coregistering(coregistered_ds):
    """Plot coregistered dataset with Master and Slave Phase Measuarements."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))  # noqa RUF059
    coregistered_ds.band_data.sel(band=1).plot(ax=axes[0], cmap="gray", robust=True)
    axes[0].set_title("Master Phase Measurement - 28 Jun 2019")

    coregistered_ds.band_data.sel(band=2).plot(ax=axes[1], cmap="gray", robust=True)
    axes[1].set_title("Slave Phase Measurement - 10 Jul 2019")

    plt.tight_layout()


def plot_interferogram(interferogram_ds):
    """Plot using interferogram dataset."""
    interferogram_ds = interferogram_ds.where(interferogram_ds != 0)
    igf_da = interferogram_ds.sel(band=1).band_data
    coh_da = interferogram_ds.sel(band=2).band_data

    # Invert y axis
    igf_da["y"] = igf_da.y[::-1]
    coh_da["y"] = coh_da.y[::-1]

    igf_plot = igf_da.hvplot.image(
        x="x",
        y="y",
        cmap=cmap_hls_hex,
        width=600,
        height=600,
        dynamic=False,
    )

    coh_plot = coh_da.hvplot.image(
        x="x",
        y="y",
        cmap="viridis",
        width=600,
        height=600,
        dynamic=False,
    ).opts(clim=(0, 1))

    return (igf_plot + coh_plot).opts(shared_axes=True)


def plot_topographic_phase_removal(interferogram_ds, topo_ds):
    """Plot topographic phase removal."""
    igf_da = interferogram_ds.sel(band=1).band_data

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # noqa RUF059

    igf_da.plot(ax=axes[0], cmap=cmap_hls)
    axes[0].set_title("Interferogram With Topographic Phase")

    topo_ds.topo.plot(ax=axes[1], cmap="gist_earth")
    axes[1].set_title("Topography")

    topo_ds.Phase.plot(ax=axes[2], cmap=cmap_hls)
    axes[2].set_title("Interferogram Without Topographic Phase")

    plt.tight_layout()


def plot_igf_coh(geocoded_ds, step):
    """Plot igf and coh data."""
    geocoded_ds = geocoded_ds.where(geocoded_ds != 0)
    igf_data = geocoded_ds.sel(band=1).band_data
    coh_da = geocoded_ds.sel(band=2).band_data

    igf_plot = igf_data.isel(x=slice(0, -1, step), y=slice(0, -1, step)).hvplot.image(
        x="x", y="y", cmap=cmap_hls_hex, width=600, height=600, dynamic=False
    )

    coh_plot = (
        coh_da.isel(x=slice(0, -1, step), y=slice(0, -1, step))
        .hvplot.image(
            x="x", y="y", cmap="viridis", width=600, height=600, dynamic=False
        )
        .opts(clim=(0, 1))
    )

    return (igf_plot + coh_plot).opts(shared_axes=True)


def array_to_img(data_array, cmap):
    """Get array as input, generate plot and save it as image in png format."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=600)
    data_array.plot(ax=ax, cmap=cmap, add_colorbar=False, add_labels=False)
    ax.set_axis_off()
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def plot_earthquake(geocoded_ds, step):
    """Plot earthquake having as input geo coded dataset."""
    igf_data = geocoded_ds.sel(band=1).band_data
    igf_data_subset = igf_data.isel(x=slice(0, -1, step), y=slice(0, -1, step))

    igf_image = array_to_img(igf_data_subset, cmap=cmap_hls)
    bounds = [
        [float(igf_data["y"].min()), float(igf_data["x"].min())],
        [float(igf_data["y"].max()), float(igf_data["x"].max())],
    ]

    m = folium.Map(
        location=[float(igf_data["y"].mean()), float(igf_data["x"].mean())],
        zoom_start=10,
    )
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/"
            "services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles © Esri",
        name="ESRI World Imagery",
    ).add_to(m)
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/"
            "services/Reference/World_Boundaries_and_Places/"
            "MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles © Esri",
        name="ESRI Labels",
        overlay=True,
    ).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{igf_image}",
        bounds=bounds,
        opacity=0.65,
        name="Interferogram",
    ).add_to(m)
    folium.LayerControl().add_to(m)

    return m
