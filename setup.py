from setuptools import setup, find_packages

setup(
    name='mola',
    version='0.0.1',
    description='ocean sunfish',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'pysam',
        'editdistance',
        'pandas',
        'numpy',
        'scipy>=1.11',
        'pyro-ppl',
        'torch',
        'igraph',
        'matplotlib',
        'seaborn',
    ],
    entry_points={
        'console_scripts': [
            'mola = mola.bin.main:mola'
        ],
    }
)
