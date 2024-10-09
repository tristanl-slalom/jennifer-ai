from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="jennifer",
    version="0.1",
    py_modules=["main"],
    packages=find_packages(),
    install_requires=required,
    entry_points={
        "console_scripts": [
            "jennifer=main:app",
        ],
    },
)