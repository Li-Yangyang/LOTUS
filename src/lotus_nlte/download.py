import requests
import os
import stat
import sys
from pathlib import Path

from .config import *

def get_url(file_name, record_id="5716361"):
    r = requests.get(f"https://zenodo.org/api/records/{record_id}")
    download_url = [f['links']['self']  for f in r.json()['files'] if f['key'] == file_name][0]
    return download_url

def is_writable(path):
    # Ensure that it exists.
    if not os.path.exists(path):
        return False

    # If we're on a posix system, check its permissions.
    if hasattr(os, "getuid"):
        statdata = os.stat(path)
        perm = stat.S_IMODE(statdata.st_mode)
        # is it world-writable?
        if perm & 0o002:
            return True
        # do we own it?
        elif statdata.st_uid == os.getuid() and (perm & 0o200):
            return True
        # are we in a group that can write to it?
        elif (statdata.st_gid in [os.getgid()] + os.getgroups()) and (perm & 0o020):
            return True
        # otherwise, we can't write to it.
        else:
            return False

    # Otherwise, we'll assume it's writable.
    # [xx] should we do other checks on other platforms?
    return True

def default_download_dir(directory):
        """
        thanks to nltk downloader function in:
        https://github.com/nltk/nltk/blob/develop/nltk/downloader.py
        Return the directory to which packages will be downloaded by
        default.  This value can be overridden using the constructor,
        or on a case-by-case basis using the ``download_dir`` argument when
        calling ``download()``.
        On Windows, the default download directory is
        ``PYTHONHOME/lib/nltk``, where *PYTHONHOME* is the
        directory containing Python, e.g. ``C:\\Python25``.
        On all other platforms, the default directory is the first of
        the following which exists or which can be created with write
        permission: ``/usr/share/nltk_data``, ``/usr/local/share/nltk_data``,
        ``/usr/lib/nltk_data``, ``/usr/local/lib/nltk_data``, ``~/nltk_data``.
        """
        # Check if we are on GAE where we cannot write into filesystem.
        if "APPENGINE_RUNTIME" in os.environ:
            return

        # Check if we have sufficient permissions to install in a
        # variety of system-wide locations.
        if not os.path.exists(directory):
            parent_dir = Path(directory).parent
            if is_writable(parent_dir):
                os.mkdir(directory)
                return directory
        elif is_writable(directory):
            return directory


        # On Windows, use %APPDATA%
        if sys.platform == "win32" and "APPDATA" in os.environ:
            homedir = os.environ["APPDATA"]

        # Otherwise, install in the user's home directory.
        else:
            homedir = os.path.expanduser("~/")
            if homedir == "~/":
                raise ValueError("Could not find a default download directory")

        # append ".lotus_package_data" to the home directory
        directory = os.mkdir(os.path.join(homedir, ".lotus_package_data"))
        return directory

def download_file(url, download_dir, path=None, clobber=False):
    """
    thanks to: https://stackoverflow.com/questions/16694907/how-to-download-large-file-in-python-with-requests-py
    path : str
        local path to download to.
    """
    if path is None:
        local_filename = os.path.join(download_dir, url.split("/")[-1])
    else:
        local_filename = path

    if os.path.exists(local_filename) and not clobber:
        print("{} exists; not downloading.".format(local_filename))
        return local_filename

    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    with open(local_filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                # f.flush() commented by recommendation from J.F.Sebastian
    return local_filename
