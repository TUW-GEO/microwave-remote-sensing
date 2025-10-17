"""Functions to process data."""

import contextlib
import os
from pathlib import Path

import cmcrameri as cmc  # noqa: F401
import numpy as np
import snaphu
import xarray as xr


@contextlib.contextmanager
def suppress_output():
    with Path.open(os.devnull, "w") as devnull:
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


def unwrap_array(  # noqa D417
    data: xr.DataArray,
    complex_var: str = "cmplx",
    ouput_var: str = "unwrapped",
    mask: xr.DataArray = True,  # noqa FBT002
    coherence: xr.DataArray = None,
    mask_nodata_value: int = 0,
    coh_low_threshold: float | None = None,
    coh_high_threshold: float | None = None,
    nlooks=1.0,
    cost="smooth",
    init="mcf",
    **kwargs,
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

    Returns
    -------
    xarray DataArray with the unwrapped phase

    """
    # Get the complex data
    data_arr = data[complex_var]

    # Create a mask for areas with no data
    if mask:
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
        coherence = np.ones_like(data_arr.real)

    # Unwrap the phase (already in complex form)
    with suppress_output():
        unw, _ = snaphu.unwrap(
            data_arr,
            coherence,
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


def subsetting(ds, x0: int = 0, y0: int = 0, dx: int = 500, dy: int = 500):
    return ds.isel(x=slice(x0, x0 + dx), y=slice(y0, y0 + dy))
