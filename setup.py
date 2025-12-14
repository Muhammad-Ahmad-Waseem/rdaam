"""
Setup configuration for r-daam package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = requirements_path.read_text().splitlines()
    requirements = [r.strip() for r in requirements if r.strip() and not r.startswith('#')]
else:
    # Fallback to minimal requirements
    requirements = [
        'torch>=2.0.0',
        'diffusers>=0.21.0',
        'transformers>=4.30.0',
        'numpy',
        'matplotlib',
        'pillow',
        'tqdm',
    ]

setup(
    name="rdaam",
    version="0.1.0",
    author="DAAM Analysis Team",
    description="Reverse Deep Attention Attribution Maps for Stable Diffusion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["scripts", "results", "tests"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'ui': ['gradio>=3.50.0'],
        'dev': ['pytest', 'flake8', 'mypy'],
    },
    entry_points={
        'console_scripts': [
            'rdaam-reverse=analysis.reverse:main',
            'rdaam-forward=analysis.forward:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
