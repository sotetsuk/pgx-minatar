from setuptools import find_packages, setup
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text()

setup(
    name="pgx-minatar",
    version="0.1.0",
    description="MinAtar extension of Pgx",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sotetsuk/pgx-minatar",
    author="Sotetsu KOYAMADA",
    author_email="sotetsu.koyamada@gmail.com",
    keywords="",
    packages=find_packages(),
    package_data={
        "": ["LICENSE"]
    },
    include_package_data=True,
    install_requires=[
        "jax>=0.3.25",  # JAX version on Colab (TPU)
        "typing_extensions",
        "pgx",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
