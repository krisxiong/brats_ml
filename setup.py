from setuptools import setup, find_packages

setup(
    name="brats-meta-learning",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "nibabel>=3.2.0",
        "pyyaml>=5.4.0",
    ],
    python_requires=">=3.7",
)
