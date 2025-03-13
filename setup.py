from setuptools import setup, find_packages

setup(
    name="clustervista",
    version="1.0",
    description="package that extracts data speed and travel time, process it, cluster it and makes visualizations, also produces an internal evaluation of the clusters ",
    author= "Daniel Chaves",
    packages=find_packages(),
    install_requires=[
        "matplotlib==3.10.1",
        "numpy==2.2.3",
        "orjson==3.10.15",
        "pandas==2.2.3",
        "pytest==8.3.5",
        "python_dateutil==2.9.0.post0",
        "scikit_learn==1.6.1",
        "setuptools==75.8.2",
    ],
)
