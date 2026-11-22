# setup.py
from setuptools import setup, find_packages

setup(
    name="SL_GPsim",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "spectral-connectivity"
    ],
    python_requires=">=3.10",
    description="Gaussian simulation and bootstrap inference from univariate and multichannel spectra",
    author="Patrick F. Bloniasz",
    author_email="patrick.bloniasz@gmail.com",
    url="https://github.com/Stephen-Lab-BU/SL_GPsim",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
