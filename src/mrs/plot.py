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
cmap_hls = sns.hls_palette(as_cmap=True)
cmap_hls_hex = sns.color_palette("hls", n_colors=256).as_hex()


def handles_(color_mapping, present_landcover_codes):
    """Generate matplotlib legend handles for present land cover codes.

    Parameters
    ----------
    color_mapping : dict
        Maps land cover IDs to dicts with 'color', 'value', and 'label'.
    present_landcover_codes : iterable
        Land cover codes to include in the legend.

    Returns
    -------
    list of matplotlib.patches.Patch
        Legend handles for the specified land cover types.

    """
    return [
        mpatches.Patch(
            color=info["color"], label=(f"{info['value']} - " + (info["label"]))
        )
        for info in color_mapping.values()
        if info["value"] in present_landcover_codes
    ]


def plot_corine_data(cor_da, cmap, norm, color_mapping, present_landcover_codes):
    """Plot CORINE land cover data with a legend.

    Parameters
    ----------
    cor_da : xarray.DataArray
        Land cover data array to plot.
    cmap : matplotlib.colors.Colormap
        Colormap used for visualization.
    norm : matplotlib.colors.Normalize
        Normalization applied to the colormap.
    color_mapping : dict
        Maps land cover IDs to dicts with 'color', 'value', and 'label'.
    present_landcover_codes : iterable
        Land cover codes to include in the legend.

    Returns
    -------
    None
        Displays the plot with CORINE data, equal aspect ratio, and a legend
        showing only present land cover classes. Includes title with EPSG code.

    """
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
    """Generate bin edges with 0.5 intervals between robust min and max values.

    Parameters
    ----------
    robust_min : float or int
        Lower bound for bin generation.
    robust_max : float or int
        Upper bound for bin generation.

    Returns
    -------
    list of float
        Sequence of bin edges spaced by 0.5 units.

    """
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
    """Create Holoviews image options based on robust intensity range.

    Parameters
    ----------
    var_ds : xarray.Dataset
        Dataset containing 'sig0' variable used to compute intensity bounds.

    Returns
    -------
    hv.opts.Image
        Holoviews image options with dynamic color limits and styling.

    """
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
    """Plot temporal variability of backscatter across land cover types.

    Parameters
    ----------
    color_mapping : dict
        Maps land cover codes to dicts with 'color', 'value', and 'label'.
    var_ds : xarray.Dataset
        Dataset containing the 'sig0' variable with time and land cover dimensions.
    present_landcover_codes : iterable
        Land cover codes to include in the analysis.

    Returns
    -------
    hv.DynamicMap
        Interactive Holoviews dynamic map showing temporal variability
        with corresponding histograms.

    """
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
    """Plot multiple Single Look Complex (SLC) intensity datasets side by side.

    Parameters
    ----------
    datasets : list of xarray.Dataset
        List of datasets, each containing an 'intensity' variable to be plotted.

    Returns
    -------
    None
        Creates a matplotlib figure with three subplots showing SLC images
        in grayscale (0-255 range), shared y-axis, and a horizontal colorbar.

    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 7), sharey=True)

    val_range = dict(vmin=0, vmax=255, cmap="gray")  # noqa C408

    for i, ds in enumerate(datasets):
        im = ds.intensity.plot(ax=ax[i], add_colorbar=False, **val_range)
        ax[i].tick_params(axis="both", which="major")

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.9, pad=0.2)  # noqa F841

    plt.show()


def plot_slc_iw2(iw2_ds):
    """Plot intensity and phase measurements for the IW2 subswath.

    Parameters
    ----------
    iw2_ds : xarray.Dataset
        Dataset containing 'intensity' and 'phase' variables for IW2.

    Returns
    -------
    None
        Creates a matplotlib figure with two subplots:
        - Left: Intensity (grayscale, robustly scaled)
        - Right: Phase (using `cmap_hls` colormap)

    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # noqa RUF059

    iw2_ds.intensity.plot(ax=axes[0], cmap="gray", robust=True)
    axes[0].set_title("Intensity Measurement of IW2")

    iw2_ds.phase.plot(ax=axes[1], cmap=cmap_hls)
    axes[1].set_title("Phase Measurement of IW2")

    plt.tight_layout()


def plot_coregistering(coregistered_ds):
    """Plot master and slave phase measurements from a coregistered dataset.

    Parameters
    ----------
    coregistered_ds : xarray.Dataset
        Dataset containing 'band_data' with at least two bands:
        band 1 for the master and band 2 for the slave measurement.

    Returns
    -------
    None
        Creates a matplotlib figure with two subplots:
        - Left: Intensity (grayscale, robustly scaled)
        - Right: Phase (using `cmap_hls` colormap)

    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))  # noqa RUF059
    coregistered_ds.band_data.sel(band=1).plot(ax=axes[0], cmap="gray", robust=True)
    axes[0].set_title("Master Phase Measurement - 28 Jun 2019")

    coregistered_ds.band_data.sel(band=2).plot(ax=axes[1], cmap="gray", robust=True)
    axes[1].set_title("Slave Phase Measurement - 10 Jul 2019")

    plt.tight_layout()


def plot_interferogram(interferogram_ds):
    """Plot interferogram and coherence data side-by-side.

    Parameters
    ----------
    interferogram_ds : xarray.Dataset
        Dataset containing interferogram and coherence bands. Expected to have
        'band_data' variable with band=1 (interferogram) and band=2 (coherence).

    Returns
    -------
    hv.Layout
        HoloViews layout with interferogram (left) and coherence (right) plots.
        Y-axis is inverted for both plots, and axes are shared.

    """
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
    """Plot interferogram before and after topographic phase removal.

    Parameters
    ----------
    interferogram_ds : xarray.Dataset
        Dataset containing the interferogram with topographic phase. Expected to
        have 'band_data' variable with band=1.
    topo_ds : xarray.Dataset
        Dataset containing topography data and the corrected interferogram.
        Expected to have 'topo' and 'Phase' data variables.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with three subplots: original interferogram, topography, and
        corrected interferogram.

    """
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
    """Plot downsampled interferogram and coherence data.

    Parameters
    ----------
    geocoded_ds : xarray.Dataset
        Dataset containing interferogram and coherence bands. Expected to have
        'band_data' variable with band=1 (interferogram) and band=2 (coherence).
    step : int
        Step size for downsampling the data along x and y axes.

    Returns
    -------
    hv.Layout
        HoloViews layout with downsampled interferogram (left) and coherence (right)
        plots. Coherence plot is scaled between 0 and 1, and axes are shared.

    """
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
    """Convert a DataArray to a base64-encoded PNG image.

    Parameters
    ----------
    data_array : xarray.DataArray
        2D data array to plot as an image.
    cmap : str
        Colormap name to use for the plot.

    Returns
    -------
    str
        Base64-encoded string of the PNG image.

    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=600)
    data_array.plot(ax=ax, cmap=cmap, add_colorbar=False, add_labels=False)
    ax.set_axis_off()
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def plot_earthquake(geocoded_ds, step):
    """Create a Folium map with a downsampled interferogram overlay.

    Parameters
    ----------
    geocoded_ds : xarray.Dataset
        Dataset containing the interferogram. Expected to have 'band_data'
        variable with band=1 and 'x'/'y' coordinates.
    step : int
        Step size for downsampling the interferogram before plotting.

    Returns
    -------
    folium.Map
        Interactive Folium map with:
        - ESRI World Imagery basemap
        - ESRI place labels overlay
        - Interferogram overlay clipped to data bounds
        - Layer control for toggling overlays

    """
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
