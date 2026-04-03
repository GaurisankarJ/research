"""Training distribution (re-search-training): vLLM + Qwen3 stack.

install_requires matches requirements-training.txt (flash-attn is excluded: pip cannot
build it in isolation; use requirements-training-flashattn.txt after pip install -r).

Recommended:
  pip install -r requirements-training.txt && pip check
  MAX_JOBS=8 pip install --no-build-isolation --no-cache-dir -r requirements-training-flashattn.txt
  python setup_training.py develop --no-deps
"""

import os
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README-training.md").read_text()

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))
with open(os.path.join(version_folder, "src/version")) as f:
    __version__ = f.read().strip()


def _read_requirements(path: str) -> list[str]:
    req_path = this_directory / path
    lines = req_path.read_text().splitlines()
    reqs: list[str] = []
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if line:
            reqs.append(line)
    return reqs


setup(
    name="re-search-training",
    version=__version__,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=_read_requirements("requirements-training.txt"),
    package_data={"": ["**/*.yaml"]},
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
