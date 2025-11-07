from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="emseg",
    version="0.1.0",
    packages=find_packages(),  # automatically finds 'emseg', submodules, etc.

    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "matplotlib>=3.6",
        "opencv-python>=4.7",
        "tensorflow>=2.9",
        "scikit-image>=0.21",
        "scikit-learn>=1.1",
        "scipy>=1.10",
        "imageio>=2.25",
        "tqdm>=4.64",
    ],

    author="Mobina Azimi",
    author_email="mobinaazimi999@gmail.com",

    description="Time-Dependent Segmentation of Electron Microscopy Images Using Deep Learning and Image Processing Methods",
    long_description=long_description,
    long_description_content_type="text/markdown",

    url="https://github.com/Mobinaazzm/EM_Segment",

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    python_requires=">=3.8",
)
