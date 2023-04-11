from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.2'
DESCRIPTION = 'Qpai'
LONG_DESCRIPTION = 'Qpai-Quantum Powered A.I. is an open source library which will be used to develop modern edge Quantum Machine Learning projects.This version includes Vision Transformer model.'

# Setting up
setup(
    name="Qpai",
    version=VERSION,
    author="Aniket Guchhait",
    author_email="aniket.emailme@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['transformer-implementations','numpy','torchvision'],
    keywords=['quantum', 'machine learning', 'quantum computing', 'qiskit', 'quantum machine learning'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)