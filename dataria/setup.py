from setuptools import setup, find_packages

setup(
    name='dataria',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'SPARQLWrapper',
        'geopandas',
        'shapely',        
    ],
    python_requires='>=3.6',   
    author='Christoph Sander',
    description='DATAria utils',
)
