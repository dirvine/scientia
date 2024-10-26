from setuptools import setup, find_packages

setup(
    name="scientia",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "click",
        "streamlit",
        # ... other dependencies from pyproject.toml
    ],
    entry_points={
        "console_scripts": [
            "scientia=scientia.cli:main",
        ],
    },
    python_requires=">=3.10",
)
