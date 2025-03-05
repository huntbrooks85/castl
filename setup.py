# setup.py
from setuptools import setup, find_packages

setup(
    name="castl",
    version="0.6.1",
    license="MIT License",
    packages=find_packages(),
    install_requires=[
        "emcee",
        "pathos",
        "IPython",
        "tqdm",
        "h5py",
        "astropy",
        "pandas",
        "matplotlib",
        "corner",
        "scikit-learn",
        "scipy",
        "numpy", 
        "xarray"
                    ],  # FILL THIS OUT ONCE I GET THE DEPENDENCIES!!!
    author="Hunter Brooks",
    author_email="hcb98@nau.edu",
    description="Computional Analysis of Spectral TempLates",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/huntbrooks85/castl", 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
