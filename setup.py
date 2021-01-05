from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="waymo2argo",
    version="0.0.1",
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX",
    ],
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements,
    dependency_links=["git+https://github.com/argoai/argoverse-api.git@master"],
)
