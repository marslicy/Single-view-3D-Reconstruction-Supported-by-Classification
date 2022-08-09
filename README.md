# 3DML project
## Description
This is the repository for our final project on the course "Machine Learning for 3D Geometry" at TUM. The repository contains two branches: the "main" for our network and the "baseline" for Wallace's work in 2019.

## Important Information

- Add the packages' name to *requirements.txt* when you installed new packages in the environment.
- Please execute the command `pre-commit install` right after you set up your python environment for the first time, such that the *black, isort, flake8* tools can run automatically before each commit to keep our codes' style.

## Folder Structures
- `utils`: utility function files.
- `data`: dataset class for ShapeNet for our tasks.
- `model`: the network that we have for the dataset.
- `runs`: the log files produced during experiments. Open it with Tensor Board.
- `configs`: specify path to dataset and logs when executing on different machines.

## Important Files
- `main.py`: set up the main parameters and the code for testing model
