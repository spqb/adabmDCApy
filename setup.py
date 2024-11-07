from setuptools import setup, find_packages

setup(
    name='adabmDCA',
    version='0.1.0',
    author='Lorenzo Rosset, Roberto Netti, Anna Paola Muntoni, Francesco Zamponi, Martin Weigt',
    author_email='rosset.lorenzo@gmail.com',
    description='Python implementation of Direct Coupling Analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/spqb/adabmDCA',
    packages=find_packages(),
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
        'matplotlib==3.9.2',
        'numpy==1.24.4',
        'pandas==2.2.3',
        'torch==2.3.0',
        'tqdm==4.66.4',
    ],
)