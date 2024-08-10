# setup.py
from setuptools import setup, find_packages

setup(
    name="castl",
    version="0.1",
    license="MIT License",
    packages=find_packages(),
    install_requires=[],  # No external dependencies for this example
    author="Hunter Brooks",
    author_email="hcb98@nau.edu",
    description="A Model Spectral Fitting MCMC Code With Linear Interpolation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/huntbrooks85/castl",  # Your GitHub repository link
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
