"""Functions for plotting."""

import base64
from collections.abc import Callable, Iterable
from functools import partial
from io import BytesIO
from typing import (
    Self,  # type: ignore[unresolved-import]
)

import folium
import holoviews as hv  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import seaborn as sns  # type: ignore[import-untyped]
from holoviews.streams import RangeXY  # type: ignore[import-untyped]
from matplotlib import patches
from matplotlib.colors import Colormap, ListedColormap, Normalize
from matplotlib.patches import Patch
from seaborn.palettes import _ColorPalette  # type: ignore[import-untyped]
from xarray import DataArray, Dataset

from mrs.catalog import CorineColorCollection, _CorineColorMapping

hv.extension("bokeh")  # type: ignore[too-many-positional-arguments]
COMPLETE_LAND_COVER = "\xa0\xa0\xa0 Complete Land Cover"
LAND_COVER: dict[str, int] = {COMPLETE_LAND_COVER: 1}
RANGEXY = RangeXY()
HIST_OPTS = hv.opts.Histogram(width=350, height=555)
CMAP_HLS: ListedColormap = sns.hls_palette(as_cmap=True)
CMAP_HLS_HEX: _ColorPalette = sns.color_palette("hls", n_colors=256).as_hex()


# A bunch of Supersets for type hinting purposes...


class _DatasetHasLandCoverSig0(Dataset):
    __slots__ = ("dataset",)
    land_cover: DataArray
    sig0: DataArray


class _DatasetHasIntensity(Dataset):
    __slots__ = ("dataset",)
    intensity: DataArray


class _DatasetHasIntensityPhase(Dataset):
    __slots__ = ("dataset",)
    intensity: DataArray
    phase: DataArray


class _DatasetHasBandData(Dataset):
    __slots__ = ("dataset",)
    band_data: DataArray


class _DatasetHasTopoPhase(Dataset):
    __slots__ = ("dataset",)
    topo: DataArray
    Phase: DataArray


class _DatasetHasPhase(Dataset):
    __slots__ = ("dataset",)
    phase: DataArray
    imag: Self
    real: Self


class _DatasetHasUnwrapped(Dataset):
    __slots__ = ("dataset",)
    unwrapped: DataArray
    phase: DataArray
    unwrapped_coh: DataArray


def _handles(
    colors: list[_CorineColorMapping],
    valid_codes: Iterable[int],
) -> list[Patch]:
    """Generate matplotlib legend handles for present land cover codes.

    Parameters
    ----------
    colors : list[CorineColorMapping]
        Maps land cover IDs to dicts with 'color', 'value', and 'label'.
    valid_codes: Iterable[int]
        Land cover codes to include in the legend.

    Returns
    -------
    list[mpl.patches.Patch]
        Legend handles for the specified land cover types.

    """
    return [
        Patch(
            color=info["color"].as_hex(),
            label=(f"{info['value']} - " + (info["label"])),
        )
        for info in colors
        if info["value"] in valid_codes
    ]


def plot_corine_data(
    cor_da: DataArray,
    cmap: Colormap,
    norm: Normalize,
    color_mapping: CorineColorCollection,
    present_landcover_codes: Iterable[int],
) -> None:
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
        figsize=(10, 10),
        cmap=cmap,
        norm=norm,
        add_colorbar=False,
    ).axes.set_aspect("equal")  # type: ignore[call-arg]

    handles = _handles(color_mapping.items, present_landcover_codes)
    plt.legend(
        handles=handles,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=7,
    )
    plt.title("CORINE Land Cover (EPSG:27704)")


def _bin_edges(robust_min: float, robust_max: float) -> list[float]:
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


def load_image(
    time: pd.Timestamp,
    land_cover: str,
    x_range: np.ndarray,
    y_range: np.ndarray,
    var_ds: _DatasetHasLandCoverSig0,
) -> hv.Image:
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
    if land_cover == COMPLETE_LAND_COVER:
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


def image_opts_(var_ds: Dataset) -> hv.opts.Image:
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


def plot_variability_over_time(
    color_mapping: dict[int, _CorineColorMapping],
    var_ds: _DatasetHasLandCoverSig0,
    present_landcover_codes: Iterable[int],
) -> hv.DynamicMap:
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
    robust_min: float = var_ds.sig0.quantile(0.02).item()
    robust_max: float = var_ds.sig0.quantile(0.98).item()

    bin_edges = _bin_edges(robust_min, robust_max)

    land_cover = LAND_COVER
    land_cover.update(
        {
            f"{int(value): 02} {color_mapping[value]['label']}": int(value)
            for value in present_landcover_codes
        },
    )
    time = var_ds.sig0["time"].values  # noqa: PD011

    load_image_partial = partial(load_image, var_ds=var_ds)

    dmap = (
        hv.DynamicMap(
            load_image_partial,
            kdims=["Time", "Landcover"],
            streams=[RANGEXY],
        )
        .redim.values(Time=time, Landcover=land_cover)
        .hist(normed=True, bins=bin_edges)
    )

    image_opts = image_opts_(var_ds)

    return dmap.opts(image_opts, HIST_OPTS)


def plot_slc_all(datasets: list[_DatasetHasIntensity]) -> None:
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

    val_range: dict[str, int | str] = {"vmin": 0, "vmax": 255, "cmap": "gray"}

    for i, ds in enumerate(datasets):
        im = ds.intensity.plot(ax=ax[i], add_colorbar=False, **val_range)  # type: ignore[arg-type]
        ax[i].tick_params(axis="both", which="major")

    fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.9, pad=0.2)
    plt.show()


def plot_slc_iw2(iw2_ds: _DatasetHasIntensityPhase) -> None:
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
    _fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    iw2_ds.intensity.plot(ax=axes[0], cmap="gray", robust=True)  # type: ignore[call-arg]
    axes[0].set_title("Intensity Measurement of IW2")

    iw2_ds.phase.plot(ax=axes[1], cmap=CMAP_HLS)  # type: ignore[call-arg]
    axes[1].set_title("Phase Measurement of IW2")

    plt.tight_layout()


def plot_coregistering(coregistered_ds: _DatasetHasBandData) -> None:
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
    _fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    coregistered_ds.band_data.sel(band=1).plot(ax=axes[0], cmap="gray", robust=True)  # type: ignore[call-arg]
    axes[0].set_title("Master Phase Measurement - 28 Jun 2019")

    coregistered_ds.band_data.sel(band=2).plot(ax=axes[1], cmap="gray", robust=True)  # type: ignore[call-arg]
    axes[1].set_title("Slave Phase Measurement - 10 Jul 2019")

    plt.tight_layout()


def plot_interferogram(interferogram_ds: _DatasetHasBandData) -> hv.Layout:
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
        cmap=CMAP_HLS_HEX,
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


def plot_topographic_phase_removal(
    interferogram_ds: _DatasetHasBandData,
    topo_ds: _DatasetHasTopoPhase,
) -> None:
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

    _fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    igf_da.plot(ax=axes[0], cmap=CMAP_HLS)  # type: ignore[call-arg]
    axes[0].set_title("Interferogram With Topographic Phase")

    topo_ds.topo.plot(ax=axes[1], cmap="gist_earth")  # type: ignore[call-arg]
    axes[1].set_title("Topography")

    topo_ds.Phase.plot(ax=axes[2], cmap=CMAP_HLS)  # type: ignore[call-arg]
    axes[2].set_title("Interferogram Without Topographic Phase")

    plt.tight_layout()


def plot_igf_coh(geocoded_ds: _DatasetHasBandData, step: int) -> hv.Layout:
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
        x="x",
        y="y",
        cmap=CMAP_HLS_HEX,
        width=600,
        height=600,
        dynamic=False,
    )

    coh_plot = (
        coh_da.isel(x=slice(0, -1, step), y=slice(0, -1, step))
        .hvplot.image(
            x="x",
            y="y",
            cmap="viridis",
            width=600,
            height=600,
            dynamic=False,
        )
        .opts(clim=(0, 1))
    )

    return (igf_plot + coh_plot).opts(shared_axes=True)


def array_to_img(data_array: DataArray, cmap: ListedColormap) -> str:
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
    data_array.plot(ax=ax, cmap=cmap, add_colorbar=False, add_labels=False)  # type: ignore[call-arg]
    ax.set_axis_off()
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def plot_earthquake(geocoded_ds: _DatasetHasBandData, step: int) -> folium.Map:
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

    igf_image = array_to_img(igf_data_subset, cmap=CMAP_HLS)
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

    folium.raster_layers.ImageOverlay(  # type: ignore[unresolved-attribute]
        image=f"data:image/png;base64,{igf_image}",
        bounds=bounds,
        opacity=0.65,
        name="Interferogram",
    ).add_to(m)
    folium.LayerControl().add_to(m)

    return m


def plot_interferogram_map(
    ds: _DatasetHasPhase,
    mask: DataArray,
    cmap_cyc: Colormap,
) -> None:
    """Plot a wrapped phase interferogram image.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset containing the 'phase' data to plot.
    mask : xarray.DataArray or array-like
        Boolean mask. Pixels where mask is False are hidden.
    cmap_cyc : matplotlib.colors.Colormap or str
        A cyclic colormap for the wrapped phase.

    Returns
    -------
    None
        Displays the plot and does not return a value.

    """
    _fig, axs = plt.subplots(figsize=(6, 6))

    (
        ds.phase.where(mask)
        .plot.imshow(cmap=cmap_cyc, zorder=1, ax=axs)
        .axes.set_title("Phase Interferogram Image (Wrapped)")
    )
    plt.show()


def plot_compare_wrapped_unwrapped_completewrapped(  # noqa: PLR0913
    subset: _DatasetHasUnwrapped,
    cmap_cyc: Colormap | str,
    ds: _DatasetHasPhase,
    mask: DataArray,
    p0: tuple[int, int],
    dxy: tuple[int, int],
) -> None:
    """Compare a subset's wrapped/unwrapped phase with its location in the full image.

    Creates a three-panel plot: the wrapped phase of a subset, its unwrapped phase,
    and the complete wrapped phase image with a rectangle indicating the subset's
    position.

    Parameters
    ----------
    subset : xarray.Dataset or xarray.DataArray
        The subset of the data, containing 'phase' and 'unwrapped' variables.
    cmap_cyc : matplotlib.colors.Colormap or str
        Cyclic colormap for the wrapped phase plots.
    ds : xarray.Dataset or xarray.DataArray
        The full dataset from which the subset was taken.
    mask : xarray.DataArray or array-like
        Boolean mask to apply to the full image plot.
    p0 : tuple[int,int]
        Top-left corner index (x,y) of the subset in the full dataset's coordinates.
    dxy : tuple[int, int]
        Width and height of the subset in number of pixels.

    Returns
    -------
    None
        Displays the plot and does not return a value.

    """
    _fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    # Wrapped Phase
    (
        subset.phase.plot.imshow(cmap=cmap_cyc, ax=axs[0]).axes.set_title(
            "Wrapped Phase of the Subset",
        )
    )

    # Unwrapped Phase
    (
        subset.unwrapped.plot.imshow(
            cmap=cmap_cyc,
            ax=axs[1],
            vmin=-80,
            vmax=80,
        ).axes.set_title("Unwrapped Phase of the Subset")
    )

    # Subset inside the complete image
    (
        ds.phase.where(mask)
        .plot.imshow(cmap=cmap_cyc, zorder=1, ax=axs[2])
        .axes.set_title("Complete Wrapped Phase Image")
    )
    x0, y0 = p0
    dx, dy = dxy
    x_start = ds.phase.coords["x"][x0].item()
    y_start = ds.phase.coords["y"][y0].item()
    x_end = ds.phase.coords["x"][x0 + dx].item()
    y_end = ds.phase.coords["y"][y0 + dy].item()

    rect = patches.Rectangle(
        (x_start, y_start),
        x_end - x_start,
        y_end - y_start,
        linewidth=1,
        edgecolor="r",
        facecolor="red",
        alpha=0.5,
        label="Subset",
    )

    # Add the rectangle to the plot
    axs[2].add_patch(rect)
    axs[2].legend()
    plt.tight_layout()


def plot_compare_coherence_mask_presence(
    subset: _DatasetHasUnwrapped,
    cmap_cyc: Colormap | str,
    threshold: float,
) -> None:
    """Compare unwrapped phase with and without a coherence threshold mask.

    Creates a two-panel plot showing the effect of applying a coherence-based
    mask to the unwrapped phase data.

    Parameters
    ----------
    subset : xarray.Dataset or xarray.DataArray
        The data subset containing 'unwrapped_coh' (phase with mask applied)
        and 'unwrapped' (phase without mask) variables.
    cmap_cyc : matplotlib.colors.Colormap or str
        Colormap for the phase plots.
    threshold : float
        The coherence threshold that was applied to generate the
        'unwrapped_coh' data.

    Returns
    -------
    None
        Displays the plot and does not return a value.

    """
    _fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    (
        subset.unwrapped_coh.plot.imshow(
            cmap=cmap_cyc,
            ax=axs[0],
            vmin=-80,
            vmax=80,
        ).axes.set_title(f"Unwrapped Phase with Coherence Threshold {threshold}")
    )

    (
        subset.unwrapped.plot.imshow(
            cmap=cmap_cyc,
            ax=axs[1],
            vmin=-80,
            vmax=80,
        ).axes.set_title("Unwrapped Phase without Coherence Threshold")
    )

    plt.show()


def plot_different_coherence_thresholds(
    ds_coh: _DatasetHasUnwrapped,
    ds_coh_2: _DatasetHasUnwrapped,
    cmap_cyc: Colormap | str,
    vmin: int = -80,
    vmax: int = 80,
) -> None:
    """Compare the results of applying two different coherence thresholds.

    Creates a two-panel plot to visually compare the unwrapped phase after
    masking with two different coherence thresholds.

    Parameters
    ----------
    ds_coh : xarray.Dataset or xarray.DataArray
        The data processed with a coherence threshold of 0.3.
        Must contain an 'unwrapped_coh' variable.
    ds_coh_2 : xarray.Dataset or xarray.DataArray
        The data processed with a coherence threshold of 0.5.
        Must contain an 'unwrapped_coh' variable.
    cmap_cyc : matplotlib.colors.Colormap or str
        Colormap for the phase plots.
    vmin : int
        The minimum value in both plots. (default -80)
    vmax : int
        The maximum value in both plots. (default 80)

    Returns
    -------
    None
        Displays the plot and does not return a value.

    """
    _fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    (
        ds_coh_2.unwrapped_coh.plot.imshow(
            cmap=cmap_cyc,
            ax=axs[0],
            vmin=vmin,
            vmax=vmax,
        ).axes.set_title("Coherence Threshold 0.5")
    )

    (
        ds_coh.unwrapped_coh.plot.imshow(
            cmap=cmap_cyc,
            ax=axs[1],
            vmin=vmin,
            vmax=vmax,
        ).axes.set_title("Coherence Threshold 0.3")
    )
    plt.show()


def plot_displacement_map(
    subset: DataArray,
    cmap_disp: Colormap | str,
    title: str,
) -> None:
    """Plot a displacement map.

    Displays a 2D displacement map with a colorbar and a title.

    Parameters
    ----------
    subset : xarray.DataArray
        The displacement data to be plotted.
    cmap_disp : matplotlib.colors.Colormap or str
        Colormap to use for the displacement map.
    title : str
        The title for the plot.

    Returns
    -------
    None
        Displays the plot and does not return a value.

    """
    (
        subset.plot.imshow(
            cmap=cmap_disp,
            cbar_kwargs={"label": "Meters [m]"},
        ).axes.set_title(title)
    )
    plt.show()


def plot_coarsened_image(
    lowres: _DatasetHasUnwrapped,
    cmap_cyc: Colormap | str,
) -> None:
    """Plot a coarsened (low-resolution) unwrapped phase map of the entire scene.

    Parameters
    ----------
    lowres : xarray.Dataset or xarray.DataArray
        The coarsened dataset containing the 'unwrapped' phase variable.
    cmap_cyc : matplotlib.colors.Colormap or str
        Cyclic colormap for the phase plot.

    Returns
    -------
    None
        Displays the plot and does not return a value.

    """
    (
        lowres.unwrapped.plot.imshow(cmap=cmap_cyc).axes.set_title(
            "Unwrapped Phase entire scene (coarsened)",
        )
    )
    plt.show()


def plot_summary(  # noqa: PLR0913
    subset: _DatasetHasUnwrapped,
    subset_disp: DataArray,
    lowres: _DatasetHasUnwrapped,
    lowres_disp: DataArray,
    cmap_cyc: Colormap | str,
    cmap_disp: Colormap | str,
) -> None:
    """Create a summary plot showing phase and displacement results.

    Generates a 2x2 grid of plots comparing a subset and the full, coarsened
    scene. The top row shows the wrapped phase, and the bottom row shows the
    corresponding displacement maps.

    Parameters
    ----------
    subset : xarray.Dataset or xarray.DataArray
        The subset of the data containing a 'unwrapped_coh' variable.
    subset_disp : xarray.Dataset or xarray.DataArray
        The displacement map for the subset.
    lowres : xarray.Dataset or xarray.DataArray
        The coarsened (low-resolution) full dataset containing a
        'unwrapped' variable.
    lowres_disp : xarray.Dataset or xarray.DataArray
        The displacement map for the coarsened full scene.
    cmap_cyc : matplotlib.colors.Colormap or str
        Cyclic colormap for the phase plots.
    cmap_disp : matplotlib.colors.Colormap or str
        Colormap for the displacement plots.

    Returns
    -------
    None
        Displays the plot and does not return a value.

    """
    _fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax = axs.ravel()

    (
        subset.unwrapped_coh.plot.imshow(
            cmap=cmap_cyc,
            ax=ax[0],
            vmin=-80,
            vmax=80,
        ).axes.set_title("Unwrapped Phase of the subset with Coherence Threshold 0.3")
    )

    (
        subset_disp.plot.imshow(
            cmap=cmap_disp,
            ax=ax[1],
            cbar_kwargs={"label": "Meters [m]"},
        ).axes.set_title("Displacement Map of the Subset")
    )

    (
        lowres.unwrapped.plot.imshow(cmap=cmap_cyc, ax=ax[2]).axes.set_title(
            "Unwrapped Phase of the entire scene with "
            "Coherence Threshold 0.3 (coarsened)",
        )
    )

    (
        lowres_disp.plot.imshow(
            cmap=cmap_disp,
            ax=ax[3],
            cbar_kwargs={"label": "Meters [m]"},
        ).axes.set_title("Displacement Map entire scene (coarse resolution)")
    )

    plt.tight_layout()


def add_histogram_to_axis(
    ax: np.ndarray,
    data: np.ndarray,
    title: str,
    max_freq: float,
    bins: int = 25,
) -> None:
    """Plot histogram on given axis with fixed settings.

    Parameters
    ----------
    ax : np.ndarray
        Matplotlib axis to plot on.
    data : np.ndarray
        Flattened data to histogram.
    title : str
        Plot title.
    max_freq : float
        Maximum frequency for y-axis limit.
    bins : int, optional
        Number of bins (default 25).

    """
    ax.hist(data.ravel(), bins=bins, range=(-20, 0), color="gray", alpha=0.7)
    ax.set_ylim(0, max_freq)
    ax.set_title(title)
    ax.set_xlabel("Backscatter (dB)")
    ax.set_ylabel("Frequency")


def plot_histograms_speckled_and_ideal_data(
    ideal_data_db: np.ndarray,
    speckled_data_db: np.ndarray,
    speckle_fraction: float,
) -> None:
    """Plot ideal and speckled data with their histograms.

    Creates a 2x2 subplot showing:
    1. Ideal data image
    2. Speckled data image
    3. Histogram of ideal data
    4. Histogram of speckled data

    Parameters
    ----------
    ideal_data_db : np.ndarray
        Ideal backscatter data in dB.
    speckled_data_db : np.ndarray
        Speckled version of the backscatter data.
    speckle_fraction : float
        Fraction of pixels affected by speckle (0-1).

    Returns
    -------
    plt.Figure
        The created figure containing all subplots.

    """
    bins = 25
    common = {"bins": bins, "range": (-20, 0)}

    hist_ideal, bins_ideal = np.histogram(ideal_data_db.ravel(), **common)  # noqa: RUF059
    hist_speckled, bins_speckled = np.histogram(speckled_data_db.ravel(), **common)  # noqa: RUF059

    # maximum frequency for normalization
    max_freq = max(hist_ideal.max(), hist_speckled.max())
    # Ideal data
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
    pos = axs[0, 0].imshow(ideal_data_db, cmap="gray", vmin=-20, vmax=0)
    axs[0, 0].set_title("Ideal Backscatter (Cornfield)")
    fig.colorbar(pos, label="Backscatter (dB)", ax=axs[0, 0])

    # Speckled data
    plt.subplot(2, 2, 2)
    pos2 = axs[0, 1].imshow(speckled_data_db, cmap="gray", vmin=-20, vmax=0)  # noqa: F841
    axs[0, 1].set_title(
        f"Speckled Backscatter ({int(speckle_fraction * 100)}% of Pixels)",
    )
    fig.colorbar(pos, label="Backscatter (dB)", ax=axs[0, 1])

    # Histogram for ideal data
    add_histogram_to_axis(
        ax=axs[1, 0],
        data=ideal_data_db,
        title="Histogram of Ideal Backscatter",
        max_freq=max_freq,
    )

    # Histogram for speckled data
    add_histogram_to_axis(
        ax=axs[1, 1],
        data=speckled_data_db,
        title=f"Histogram of Speckled Backscatter ({int(speckle_fraction * 100)}%)",
        max_freq=max_freq,
    )

    plt.tight_layout()


def load_image_landcover(  # noqa: PLR0913
    var_ds: Dataset,
    time: None,
    land_cover: any,
    x_range: DataArray,
    y_range: DataArray,
    filter_fun_spatial: None,
) -> hv.Image:
    """Load Landcover image.

    Parameters
    ----------
    var_ds: Dataset
        dataset type
    time: any
        time slice
    land_cover: any
        land cover type
    x_range: array_like
        longitude range
    y_range: array_like
        latitude range
    filter_fun_spatial: any
        filter type


    Returns
    -------
    holoviews.Image

    """
    if time is not None:
        var_ds = var_ds.sel(time=time)

    if land_cover == "\xa0\xa0\xa0 Complete Land Cover":
        sig0_selected_ds = var_ds.sig0
    else:
        land_cover_value = int(land_cover.split()[0])
        mask_ds = var_ds.land_cover == land_cover_value
        sig0_selected_ds = var_ds.sig0.where(mask_ds)

    if filter_fun_spatial is not None:
        sig0_np = filter_fun_spatial(sig0_selected_ds.to_numpy())
    else:
        sig0_np = sig0_selected_ds.to_numpy()

    # Convert unfiltered data into Holoviews Image
    img = hv.Dataset(
        (sig0_selected_ds["x"], sig0_selected_ds["y"], sig0_np),
        ["x", "y"],
        "sig0",
    )

    if x_range and y_range:
        img = img.select(x=x_range, y=y_range)

    return hv.Image(img)


def plot_variability(
    var_ds: Dataset,
    color_mapping: dict,
    present_landcover_codes: list,
    filter_fun_spatial: Callable[[Dataset], Dataset] | None = None,
    filter_fun_temporal: Callable[[Dataset], Dataset] | None = None,
) -> hv.core.layout:
    """Create an interactive plot for backscatter data exploration.

    Generates a DynamicMap showing an image of backscatter values alongside
    a synchronized histogram. The plot supports dynamic selection of time
    and landcover type via widgets.

    Parameters
    ----------
    var_ds : xr.Dataset
        Dataset containing backscatter data ('sig0' variable).
    color_mapping : dict
        Mapping of landcover codes to color and label information.
    present_landcover_codes : list
        List of landcover codes available in the data.
    filter_fun_spatial : callable, optional
        Function to spatially filter the dataset.
    filter_fun_temporal : callable, optional
        Function to temporally filter the dataset.

    Returns
    -------
    holoviews.core.Layout
        An interactive HoloViews DynamicMap with linked image and histogram.

    """
    robust_min = var_ds.sig0.quantile(0.02).item()
    robust_max = var_ds.sig0.quantile(0.98).item()

    bin_edges = [
        i + j * 0.5
        for i in range(int(robust_min) - 2, int(robust_max) + 2)
        for j in range(2)
    ]

    land_cover = {"\xa0\xa0\xa0 Complete Land Cover": 1}
    land_cover.update(
        {
            f"{int(value): 02} {color_mapping[value]['label']}": int(value)
            for value in present_landcover_codes
        },
    )
    time = var_ds.sig0["time"].to_numpy()

    rangexy = RangeXY()

    if filter_fun_temporal is not None:
        var_ds = filter_fun_temporal(var_ds)
        load_image_landcover_ = partial(
            load_image_landcover,
            var_ds=var_ds,
            filter_fun_spatial=filter_fun_spatial,
            time=None,
        )
        dmap = (
            hv.DynamicMap(load_image_landcover_, kdims=["Landcover"], streams=[rangexy])
            .redim.values(Landcover=land_cover)
            .hist(normed=True, bins=bin_edges)
        )

    else:
        load_image_landcover_ = partial(
            load_image_landcover,
            var_ds=var_ds,
            filter_fun_spatial=filter_fun_spatial,
        )
        dmap = (
            hv.DynamicMap(
                load_image_landcover_,
                kdims=["Time", "Landcover"],
                streams=[rangexy],
            )
            .redim.values(Time=time, Landcover=land_cover)
            .hist(normed=True, bins=bin_edges)
        )

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

    hist_opts = hv.opts.Histogram(width=350, height=555)

    return dmap.opts(image_opts, hist_opts)
