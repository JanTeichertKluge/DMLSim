from setuptools import find_packages, setup
setup(
    name='DMLSim',
    packages=find_packages(),
    version='0.1.0',
    description='package to run DoubleML simulation studies',
    author='Sundermann, Moritz; Teichert-Kluge, Jan',
    install_requires=['DoubleML',
                      'matplotlib',
                      'numpy',
                      'openpyxl',
                      'pandas',
                      'Pillow',
                      'python-dateutil',
                      'scikit-learn',
                      'scipy==1.7.3',
                      'seaborn',
                      'skorch',
                      'statsmodels',
                      'torch',
                      'tqdm'],
     license='MIT'
)
