import setuptools

with open("requirements.txt", "r") as fh:
    requirements = fh.read()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="arpys",
    version="0.1.0",
    author="Kevin Kramer",
    author_email="kevin.kramer@uzh.ch",
    description="Python tools and scripts for ARPES data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kuadrat/arpys.git",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
)

