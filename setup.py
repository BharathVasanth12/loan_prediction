from setuptools import setup, find_packages

def read_requirements(filename: str) -> list:
    """Read the requirements from a file and return a list of dependencies."""
    with open(filename, 'r') as file:
        return file.read().splitlines()
    
setup(
    name = 'Loan Prediction Pipeline',
    version = '0.1.0',
    packages = find_packages(),
    install_requires = read_requirements('requirements.txt'),
    description = 'A modular machine learning pipeline for loan prediction',
    author = 'Bharath Vasanthkumar',
    author_email = 'bharath.vasanthkumar@gmail.com'
)