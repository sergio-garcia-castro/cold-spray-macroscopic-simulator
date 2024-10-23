from setuptools import setup, find_packages

# This setup excludes the PyTorch3d library, you will have to install it through pip or Conda.
setup(
    name='ColdSpray',
    version='0.1.0',
    author='Sergio Garcia',
    author_email='sergio.garcia_castro@etu.minesparis.psl.eu',
    description='Macroscopic simulator of Cold Spray deposition',
    url='https://github.com/sergio-garcia-castro/cold-spray-macroscopic-simulator/',
    install_requires=["torch", 
        "torchvision", 
        "torchaudio", 
        "iopath",
        "matplotlib"],
    py_modules=[],
    python_requires='>=3.9.19')
