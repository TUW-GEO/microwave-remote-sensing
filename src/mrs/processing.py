"""Functions to process data."""

import contextlib
import os
from pathlib import Path
from typing import Literal, Never  # type: ignore[ty-not-there-yet]

import numpy as np
import snaphu
import xarray as xr


@contextlib.contextmanager
def suppress_output():  # noqa: ANN201
    """Suppress stdout and stderr temporarily with a context manager.

    This manager redirects all standard output and standard error to
    ``os.devnull`` for the duration of the ``with`` block. The original
    streams are guaranteed to be restored when the block exits, even if
    errors occur.

    Yields
    ------
    None
        Provides the context for the ``with`` statement; does not return a value.

    """
    with Path.open(os.devnull, "w") as devnull:  # type: ignore[no-overload-matching]
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)

        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)

        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)


# TODO: Remove complexity from the function  # noqa: FIX002
def unwrap_array(  # noqa: D417, PLR0913
    data: xr.DataArray,
    complex_var: str = "cmplx",
    ouput_var: str = "unwrapped",
    mask: xr.DataArray | None = None,
    coherence: xr.DataArray | None = None,
    mask_nodata_value: int = 0,
    coh_low_threshold: float | None = None,
    coh_high_threshold: float | None = None,
    nlooks: float = 1.0,
    cost: Literal["defo", "smooth"] = "smooth",
    init: Literal["mcf", "mst"] = "mcf",
    **kwargs: Never,
) -> xr.DataArray:
    """Unwraps the phase data using the snaphu algorithm.

    Parameters
    ----------
    data: xarray DataArray with complex numbers
    complex_var: Name of the variable with the complex numbers
    ouput_var: Name of the variable with the unwrapped phase
    mask: xarray DataArray with mask values
    coherence: xarray DataArray with coherence values (optional)
    mask_nodata_value: Value of the no data pixels in the mask
    coh_low_threshold: Lower threshold for the coherence values
    coh_high_threshold: Higher threshold for the coherence values
    nlooks: The number of looks
    cost: Statistical cost mode
    init: Algorithm used for initialization of unwrapped phase gradients.

    Returns
    -------
    xarray DataArray with the unwrapped phase

    """
    # Get the complex data
    data_arr = data[complex_var]

    # Create a mask for areas with no data
    if mask is None:
        mask = (data_arr.real != mask_nodata_value).astype(bool)

    # Apply coherence thresholds if provided
    if coherence is not None:
        if coh_low_threshold is not None:
            coh_mask = (coherence >= coh_low_threshold).astype(bool)
            mask = mask & coh_mask
        if coh_high_threshold is not None:
            coh_mask = (coherence <= coh_high_threshold).astype(bool)
            mask = mask & coh_mask

    # Apply the mask to the data
    data_arr = data_arr.where(mask)

    if coherence is None:
        # TODO: Resolve type-checker complaints  # noqa: FIX002
        coherence: np.ndarray = np.ones_like(data_arr.real)  # type: ignore[no-redef]

    # Unwrap the phase (already in complex form)
    with suppress_output():
        unw, _ = snaphu.unwrap(
            igram=data_arr,
            corr=coherence,  # type: ignore[arg-type]
            nlooks=nlooks,
            cost=cost,
            init=init,
            mask=mask,
            **kwargs,
        )

    # Build xarray DataArray with the unwrapped phase
    data[ouput_var] = (("y", "x"), unw)

    # Mask the unwrapped phase
    data[ouput_var] = data[ouput_var].where(mask)
    return data


def subsetting(
    ds: xr.DataArray | xr.Dataset,
    x0: int = 0,
    y0: int = 0,
    dx: int = 500,
    dy: int = 500,
) -> xr.DataArray | xr.Dataset:
    """Extract a rectangular subset from an xarray object.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The data object to subset.
    x0 : int, default 0
        Starting index along the 'x' dimension.
    y0 : int, default 0
        Starting index along the 'y' dimension.
    dx : int, default 500
        The size (number of pixels) of the subset along the 'x' dimension.
    dy : int, default 500
        The size (number of pixels) of the subset along the 'y' dimension.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        A new object containing only the data within the specified slice.

    """
    return ds.isel(x=slice(x0, x0 + dx), y=slice(y0, y0 + dy))
