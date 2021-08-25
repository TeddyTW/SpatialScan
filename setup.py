import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SpatialScan",
    version="0.1",
    description="Expectation Based Scan Statistic functionality.",
    url="https://github.com/TeddyTW/SpatialScan",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chance Haycock, Edward Thorpe-Woods",
    author_email="chancehaycock@me.com, t_thorpewoods@hotmail.co.uk",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "matplotlib==3.2.2",
        "numpy==1.18.5",
        "tensorflow==2.5.1",
        "seaborn==0.10.1",
        "plotly==4.8.1",
        "pandas==1.0.5",
        "scikit_learn==0.23.1",
    ],
)
