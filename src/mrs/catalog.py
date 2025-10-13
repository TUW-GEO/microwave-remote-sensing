"""Make URLs for Intake catalog and geotiffs.

Returns
-------
        URL for intake catalog : str
        URL for Sentinel-1 and Alos-2 geotiff : str

"""

import urllib.parse

import gitlab

ROOT = "https://git.geo.tuwien.ac.at"
REPO = "/api/v4/projects/1264/repository/files/"


def get_intake_url(root: str = ROOT, branch: str = "main"):
    """Create URL for intake catalog.

    Parameters
    ----------
    root : str
        Root of URL
    branch : str
        Branch of GitLab repository

    Returns
    -------
        Intake catalog : str

    """
    intake_path = root + REPO + f"microwave-remote-sensing.yml/raw?ref={branch}"
    print(intake_path)
    return root


def make_gitlab_urls(sensor):
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
    root = ROOT + REPO
    end = "/raw?ref=main&lfs=true"
    return [
        root + urllib.parse.quote_plus(gitlab_file["path"]) + end
        for gitlab_file in gl_project.repository_tree(
            sensor, ref="main", recursive=True, iterator=True
        )
        if gitlab_file["path"].endswith(".tif")
    ]
