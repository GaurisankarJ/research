import os
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))
with open(os.path.join(version_folder, "src/version")) as f:
    __version__ = f.read().strip()


def _read_requirements(path: str | None = None) -> list[str]:
    """Parse requirements.txt for setuptools (skip blanks and # comments)."""
    path = path or os.environ.get("RESEARCH_REQUIREMENTS_FILE", "requirements.txt")
    req_path = this_directory / path
    lines = req_path.read_text().splitlines()
    reqs: list[str] = []
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if line:
            reqs.append(line)
    return reqs


setup(
    name="re-search",
    version=__version__,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=_read_requirements(),
    package_data={"": ["**/*.yaml"]},
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
