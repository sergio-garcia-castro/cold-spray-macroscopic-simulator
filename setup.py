from setuptools import setup

setup(
    name='ColdSpray',
    version='0.1.0',  # Define your package version
    author='Sergio Garcia',  # Replace with your name
    author_email='sergio.garcia_castro@etu.minesparis.psl.eu',  # Replace with your email
    description='Macroscopic simulator of Cold Spray deposition',
    long_description=open('README.md').read(),  # You can create a README.md file for long description
    long_description_content_type='text/markdown',
    url='https://github.com/sergio-garcia-castro/cold-spray-macroscopic-simulator/',  # Replace with your project URL
    install_requires=["matplotlib", "pytorch==1.13.0", "pytorch3d=0.7.5"],  # Specify the dependencies
    python_requires='==3.9.19',  # Specify the required Python version
)

