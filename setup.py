from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sensor-localization",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Decentralized sensor network localization using MPS and ADMM algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sensor-localization",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "cvxpy>=1.2.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "mpi": ["mpi4py>=3.1.0"],
        "dev": ["pytest>=6.2.0", "pytest-mpi>=0.6"],
    },
    entry_points={
        "console_scripts": [
            "run-localization=examples.simple_example:run_simple_example",
            "compare-algorithms=examples.mps_vs_admm:run_comparison",
        ],
    },
)