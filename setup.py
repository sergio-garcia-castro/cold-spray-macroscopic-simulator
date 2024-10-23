from setuptools import setup

# Setup configuration
setup(
    name='ColdSprayTest',
    version='0.1.0',
    author='Sergio Garcia',
    author_email='sergio.garcia_castro@etu.minesparis.psl.eu',
    description='Macroscopic simulator of Cold Spray deposition',
    packages=['Cold_Spray'],
    url='https://github.com/sergio-garcia-castro/cold-spray-macroscopic-simulator/',
    install_requires=["torch", 
        "torchvision", 
        "torchaudio", 
        "iopath",
        "matplotlib"],
    py_modules=[],
    python_requires='>=3.9.19')
