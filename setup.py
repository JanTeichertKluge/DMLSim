from setuptools import find_packages, setup
setup(
    name='DMLSim',
    packages=find_packages(),
    version='0.1.0',
    description='package to run DoubleML simulation studies',
    author='Sundermann, Moritz; Teichert-Kluge, Jan',
    install_requires=['distlib,
                      'DoubleML',
                      'fonttools',
                      'joblib',
                      'jsonschema',
                      'matplotlib',
                      'more-itertools',
                      'numpy',
                      'openpyxl',
                      'pandas',
                      'patsy',
                      'pexpect,
                      'Pillow',
                      'platformdirs',
                      'pluggy',
                      'ptyprocess',
                      'pytest==7.2.0',
                      'python-dateutil',
                      'pytz',
                      'scikit-learn==1.2.0',
                      'scipy==1.7.3',
                      'seaborn',
                      'skorch',
                      'statsmodels',
                      'tabulate',
                      'torch==1.13.1',
                      'tqdm',
                      'trove-classifiers',
                      'typing_extensions']
     license='MIT'
)
