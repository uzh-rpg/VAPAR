import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

print(setuptools.find_packages(include=["gazesim"]))

setuptools.setup(
    name="gazesim",
    version="0.0.1",
    author="Simon Wengeler",
    author_email="simon.wengeler@outlook.com",
    description="Tools for working with the UZH-RPG FPV Saliency (GazeSim) dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/swengeler/fpv_saliency_maps",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[""],
)
