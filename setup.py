from setuptools import setup, find_packages

setup(
    name='adabmDCA',
    version='0.3.0',
    author='Lorenzo Rosset, Roberto Netti, Anna Paola Muntoni, Francesco Zamponi, Martin Weigt',
    maintainer='Lorenzo Rosset',
    author_email='rosset.lorenzo@gmail.com',
    description='Python implementation of Direct Coupling Analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/spqb/adabmDCA',
    packages=find_packages(include=['adabmDCA', 'adabmDCA.*']),
    include_package_data=True,
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'adabmDCA=adabmDCA.cli:main',
        ],
    },
    install_requires=[
        'matplotlib>=3.8.0',
        'numpy>=1.26.4',
        'pandas>=2.2.2',
        'torch>=2.5.1',
        'tqdm>=4.66.6',
        'h5py>=3.12.1',
        'wandb>=0.12.0',
    ],
)