"""Make URLs for Intake catalog and geotiffs.

Returns
-------
        URL for intake catalog : str
        URL for Sentinel-1 and Alos-2 geotiff : str

"""

import urllib.parse
from enum import StrEnum  # type: ignore[unresolved-import]

import gitlab

ROOT = "https://git.geo.tuwien.ac.at"
REPO_API = "api/v4/projects/1264/repository/files"
REPO_RAW = "public_projects/microwave-remote-sensing/-/raw"


class SensorOptions(StrEnum):
    """Supported Sensor Options."""

    ALOS2 = "alos-2"
    SENTINEL1 = "sentinel-1"


def get_intake_url(
    root: str = ROOT,
    branch: str = "main",
    repo: str = REPO_RAW,
    *,
    verbose: bool = True,
) -> str:
    """Create URL for an intake catalog.

    Parameters
    ----------
    root : str
        Root of URL
    branch: str
        The branch from the repo that should be accessed
    repo: str
        The repo that the intake should take the data from
    verbose: bool
        If true prints the returned path to stdout

    Returns
    -------
        Intake catalog : str

    """
    intake_path = f"{root}/{repo}/{branch}/microwave-remote-sensing.yml"
    if verbose:
        print(intake_path)
    return intake_path


def make_gitlab_urls(sensor: SensorOptions | str):
    """Create URL to Alos and Sentinel filed on GitLab.

    Parameters
    ----------
    sensor : str
        either "alos-2" or "sentinel-1"

    Returns
    -------
        URL : str

    """
    gl = gitlab.Gitlab(ROOT)
    gl_project = gl.projects.get(1264)
    root = f"{ROOT}/{REPO_API}/"
    end = "/raw?ref=main&lfs=true"
    return [
        root + urllib.parse.quote_plus(gitlab_file["path"]) + end
        for gitlab_file in gl_project.repository_tree(
            sensor,
            ref="main",
            recursive=True,
            iterator=True,
        )
        if gitlab_file["path"].endswith(".tif")
    ]
