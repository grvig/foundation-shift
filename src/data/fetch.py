"""Fetch many files from the official bundle, one request per file, in parallel.

iWildCam has no mirror, and the bundle downloads whole only as a stream that cannot
resume. Single files download fine, so each image is requested on its own by a pool of
threads. That makes the download resumable for free: files already on disk are skipped,
so a rerun after a dropped connection fetches only what is missing.

Each file is written under a temporary name and renamed once complete, and must open as
an image before it counts. A file that keeps failing after several attempts is reported
at the end rather than stopping the other threads; preparation then fails, because a
dataset with holes in it would change every number computed from it.
"""

import io
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

ATTEMPTS = 4


def default_open(url):
    return urllib.request.urlopen(url, timeout=60)


def check_image(data):
    with Image.open(io.BytesIO(data)) as image:
        image.verify()


def fetch_one(base_url, relative_path, target_dir, open_url):
    """Return None on success, or a short description of why the file failed."""
    target = target_dir / relative_path
    if target.exists():
        return None
    last_error = ""
    for attempt in range(ATTEMPTS):
        try:
            with open_url(base_url + relative_path) as response:
                data = response.read()
            check_image(data)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(target)
            return None
        except (urllib.error.URLError, OSError, ValueError) as error:
            last_error = str(error)[:120]
            time.sleep(0.5 * (attempt + 1))
    return relative_path + ": " + last_error


def fetch_files(base_url, relative_paths, target_dir, workers, open_url=None, log=print):
    """Fetch every missing file; raise if any could not be fetched."""
    if open_url is None:
        open_url = default_open
    missing = []
    for path in relative_paths:
        if not (target_dir / path).exists():
            missing.append(path)
    log(str(len(relative_paths) - len(missing)) + " files already present, "
        + str(len(missing)) + " to fetch")
    failures = []
    done = 0
    started = time.time()
    with ThreadPoolExecutor(int(workers)) as pool:
        futures = []
        for path in missing:
            futures.append(pool.submit(fetch_one, base_url, path, target_dir, open_url))
        for future in futures:
            problem = future.result()
            done = done + 1
            if problem is not None:
                failures.append(problem)
            if done % 5000 == 0:
                rate = done / max(time.time() - started, 1e-9)
                log("  " + str(done) + "/" + str(len(missing)) + " ("
                    + format(rate, ".0f") + " files/s)")
    if len(failures) > 0:
        raise ValueError(str(len(failures)) + " files could not be fetched; rerun to "
                         "retry them. First: " + failures[0])
    return len(missing)
