from setuptools import setup

__author__ = "Derek Rollend"
__version__ = "0.1"

setup(
    name="sample",
    version=__version__,
    description="Code sample package for demonstration purposes.",
    long_description=open("README.md").read(),
    author=__author__,
    author_email="derek.rollend@gmail.com",
    license="TBD",
    packages=["sample"],
    install_requires=[
        req.strip()
        for req in open("requirements.txt").read().splitlines()
        if not req.strip().startswith("#")
    ],
)
