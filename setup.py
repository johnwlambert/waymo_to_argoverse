from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# dependency_links not needed, install_requires sufficient
# per PEP 508 https://www.python.org/dev/peps/pep-0508/
# and https://stackoverflow.com/a/54216163

setup(
    name="waymo2argo",
    version="0.0.1",
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX",
    ],
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements
)
