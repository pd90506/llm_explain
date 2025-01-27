from setuptools import setup, find_packages

setup(
    name='llmexp',  # package name
    version='0.1.0', # package version
    packages=find_packages(), # find all packages in the directory
    install_requires=[ 
        # list of dependencies
    ],
    author='Deng',
    author_email='pd90506@gmail.com',
    description='A package for explaining LLMs',
    license='MIT'
)

# Install the package in editable mode
# pip install -e .