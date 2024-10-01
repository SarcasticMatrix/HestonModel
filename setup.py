from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='hestonModel',
    version='0.1.4',
    author="Théophile Schmutz",
    url="https://sarcasticmatrix.github.io/hestonModel/",
    description="Heston model for option pricing and portfolio management using Monte Carlo simulations, the Carr-Madan method, and Fourier transforms",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(where='hestonModel'),
    readme='README.md',
    package_dir={'': 'hestonModel'},
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy",
        "statsmodels",
        "tqdm",
    ],
)
