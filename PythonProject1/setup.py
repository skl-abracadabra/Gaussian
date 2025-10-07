from setuptools import setup, find_packages

setup(
    name="gaussian_image_registration",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "nibabel>=3.2.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
    ],
)
