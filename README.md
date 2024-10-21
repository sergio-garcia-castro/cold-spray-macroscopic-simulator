# Cold Spray Macroscopic simulator

## Description
This project is designed to 3D model the Cold Spray process. The method used for this was through mesh modifcation, where we use the mathematical model of Cold Spray to adequately modify the points of the mesh.
In the repository it is included the main code, several examples, and `.obj` files that are loaded and used as meshes in both the examples and the main code. 

Key features:
- Mesh handling from `.obj` files
- Implementation of the mathematical model of the Cold Spray process.
- Optimization during the simulation of the mesh modification.
- Export final meshes.

## Prerequisites
Before getting started, ensure you have the following software installed:
1. **Anaconda** or **Miniconda**: You can download and install it from [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. **Git**: To clone the repository, install Git from [here](https://git-scm.com/).

## Installation
Due to limitations of one of the packages used, PyTorch3d, this project can only be installed on a Linux environment.

### Step 1: Clone the Repository

First, clone the project repository from GitHub using the following command in your terminal:

```bash
git clone https://github.com/sergio-garcia-castro/cold-spray-modeling.git
```
Then, navigate to the project folder:
```bash
cd cold-spray-modeling
```
Use the cold_spray.yml file provided in the repository to recreate the project's virtual environment. This file contains all the required dependencies for running the project.

Create and activate the environment using the following commands:
```bash
conda env create -f cold_spray.yml
conda activate ColdSpray
```
This will install all the necessary packages.

## Usage
The code was developed in Python using the following main libraries:
\begin{itemize}
    \item \textbf{PyTorch}: for tensor computation of mathematical operations and optimization.
    \item \textbf{PyTorch3D}: for manipulating and modifying 3D meshes from simulations.
    \item \textbf{Pyvista}: for 3D visualization of meshes.
\end{itemize}

The file `cold_spray.py` contains all the classes and methods used for the simulation. The details and explanation can be found in the documentation inside the file. 
The file `main.py` contains a first demonstration on the simulation and optimization of the Cold Spray simulation. This file loads some of the meshes found in the meshes folder and permors and optimization process in order
