from setuptools import find_packages, setup

setup(
    name='tsadams',
    version='0.1.0',
    description='Unsupervised Time Series Anomaly Detection Model Selection',
    url='https://github.com/mononitogoswami/tsad-model-selection',
    author='Mononito Goswami',
    author_email='mgoswami@cs.cmu.edu',
    license='Apache 2.0',
    packages=find_packages(exclude=['tests', 'configs']),
    install_requires=['cvxopt',
                      'cvxpy',
                      'matplotlib',
                      'networkx',
                      'pandas',
                      'patool',
                      'setuptools',
                      'scikit-learn',
                      'statsmodels',
                      'tqdm',
                      ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.10',
    ],
)