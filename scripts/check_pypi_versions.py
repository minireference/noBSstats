#!/usr/bin/env python3
"""
List installed packages in the current virtualenv, compare them to the latest
versions on PyPI, and report whether each package is up to date.

Roughly analogous to `pip freeze`, but with an added PyPI version check.

Usage:
    python check_pypi_versions.py
    python check_pypi_versions.py --outdated-only
    python check_pypi_versions.py --json
    python check_pypi_versions.py --workers 16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Iterable

try:
    from packaging.version import InvalidVersion, Version
except ImportError:
    # Fallback: packaging is usually vendored inside pip
    from pip._vendor.packaging.version import InvalidVersion, Version  # type: ignore


PYPI_URL_TEMPLATE = "https://pypi.org/pypi/{package}/json"
USER_AGENT = "venv-version-checker/1.0"


@dataclass
class PackageInfo:
    name: str
    installed: str
    latest: str | None
    status: str
    error: str | None = None


def normalize_name(name: str) -> str:
    """
    PEP 503-style normalization:
    replace runs of -, _, . with single -
    and lowercase the name.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def get_installed_packages() -> list[tuple[str, str]]:
    """
    Return installed distributions as (name, version), sorted by normalized name.
    """
    packages: dict[str, str] = {}

    for dist in metadata.distributions():
        name = dist.metadata.get("Name") or dist.metadata.get("Summary")
        version = dist.version

        if not name or not version:
            continue

        # De-duplicate by normalized name
        norm = normalize_name(name)
        if norm not in packages:
            packages[norm] = version

    return sorted(packages.items(), key=lambda x: x[0])


def fetch_latest_version(package_name: str, timeout: float = 10.0) -> str:
    """
    Query PyPI JSON API for the latest version of a package.
    """
    url = PYPI_URL_TEMPLATE.format(package=urllib.parse.quote(package_name))
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.load(response)

    latest = data["info"]["version"]
    if not isinstance(latest, str):
        raise ValueError(f"Unexpected version payload for {package_name!r}")
    return latest


def compare_versions(installed: str, latest: str) -> str:
    """
    Compare installed vs latest using PEP 440 version parsing.
    """
    try:
        installed_v = Version(installed)
        latest_v = Version(latest)
    except InvalidVersion:
        if installed == latest:
            return "up-to-date"
        return "unknown"

    if installed_v == latest_v:
        return "up-to-date"
    if installed_v < latest_v:
        return "behind"
    return "ahead"


def check_one(package_name: str, installed_version: str, timeout: float) -> PackageInfo:
    try:
        latest = fetch_latest_version(package_name, timeout=timeout)
        status = compare_versions(installed_version, latest)
        return PackageInfo(
            name=package_name,
            installed=installed_version,
            latest=latest,
            status=status,
        )
    except urllib.error.HTTPError as e:
        # Common for packages not published on PyPI under that exact name
        return PackageInfo(
            name=package_name,
            installed=installed_version,
            latest=None,
            status="unavailable",
            error=f"HTTP {e.code}",
        )
    except urllib.error.URLError as e:
        return PackageInfo(
            name=package_name,
            installed=installed_version,
            latest=None,
            status="error",
            error=str(e.reason),
        )
    except Exception as e:
        return PackageInfo(
            name=package_name,
            installed=installed_version,
            latest=None,
            status="error",
            error=str(e),
        )


def check_packages(
    packages: Iterable[tuple[str, str]],
    workers: int,
    timeout: float,
) -> list[PackageInfo]:
    results: list[PackageInfo] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(check_one, name, installed, timeout): (name, installed)
            for name, installed in packages
        }

        for future in as_completed(futures):
            results.append(future.result())

    return sorted(results, key=lambda p: normalize_name(p.name))


def print_table(results: list[PackageInfo], outdated_only: bool = False) -> None:
    if outdated_only:
        results = [r for r in results if r.status == "behind"]

    if not results:
        print("No matching packages.")
        return

    name_w = max(len("PACKAGE"), max(len(r.name) for r in results))
    inst_w = max(len("INSTALLED"), max(len(r.installed) for r in results))
    latest_w = max(
        len("LATEST"),
        max(len(r.latest) if r.latest is not None else 1 for r in results),
    )
    status_w = max(len("STATUS"), max(len(r.status) for r in results))

    header = (
        f"{'PACKAGE':<{name_w}}  "
        f"{'INSTALLED':<{inst_w}}  "
        f"{'LATEST':<{latest_w}}  "
        f"{'STATUS':<{status_w}}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        latest = r.latest if r.latest is not None else "-"
        line = (
            f"{r.name:<{name_w}}  "
            f"{r.installed:<{inst_w}}  "
            f"{latest:<{latest_w}}  "
            f"{r.status:<{status_w}}"
        )
        if r.error:
            line += f"  ({r.error})"
        print(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare installed virtualenv packages against the latest PyPI versions."
    )
    parser.add_argument(
        "--outdated-only",
        action="store_true",
        help="Show only packages that are behind the latest PyPI version.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a text table.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Number of concurrent PyPI requests (default: 12).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Per-request timeout in seconds (default: 10).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packages = get_installed_packages()

    if not packages:
        print("No installed packages found.")
        return 0

    results = check_packages(packages, workers=args.workers, timeout=args.timeout)

    if args.outdated_only:
        results = [r for r in results if r.status == "behind"]

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        print_table(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())


