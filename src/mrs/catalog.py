"""Make URLs for Intake catalog and geotiffs.

Returns
-------
        URL for intake catalog : str
        URL for Sentinel-1 and Alos-2 geotiff : str

"""

import os
import urllib.parse
from enum import StrEnum  # type: ignore[unresolved-import]

import gitlab
from pydantic import BaseModel, Field
from pydantic_extra_types.color import Color
from typing_extensions import TypedDict

ROOT = "https://git.geo.tuwien.ac.at"
REPO_API = "api/v4/projects/1264/repository/files"
REPO_RAW = "public_projects/microwave-remote-sensing/-/raw"

os.environ["CACHE_DEST"] = "/tmp/fsspec"  # noqa COM812


class _SensorOptions(StrEnum):
    """Supported Sensor Options."""

    ALOS2 = "alos-2"
    SENTINEL1 = "sentinel-1"


class _CorineColorMapping(TypedDict):
    """Data Model of the expected Corine Color Mapping."""

    value: int
    color: Color
    label: str


class CorineColorCollection(BaseModel):
    """Data Model of the expected Corine Color Mapping Collection."""

    items: list[_CorineColorMapping] = Field(alias="land_cover")

    def to_dict(self) -> dict[int, _CorineColorMapping]:
        """Convert to dictionary."""
        return {item["value"]: item for item in self.items}


def get_intake_url(
    root: str = ROOT,
    branch: str = "dev-exs",
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


def make_gitlab_urls(sensor: _SensorOptions | str) -> list[str]:
    """Create URL to Alos and Sentinel filed on GitLab.

    Parameters
    ----------
    sensor : str
        either "alos-2" or "sentinel-1"

    Returns
    -------
        URL : list[str]

    """
    gl = gitlab.Gitlab(url=ROOT)
    gl_project = gl.projects.get(id=1264)
    root = f"{ROOT}/{REPO_API}/"
    end = "/raw?ref=main&lfs=true"
    return [
        root + urllib.parse.quote_plus(gitlab_file["path"]) + end
        for gitlab_file in gl_project.repository_tree(
            path=sensor,
            ref="main",
            recursive=True,
            iterator=True,
        )
        if gitlab_file["path"].endswith(".tif")
    ]
