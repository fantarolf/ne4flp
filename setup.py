import setuptools

PACKAGES = [
    'numpy==1.15.4',
    'networkx==2.2',
    'pandas==0.22.0',
    'scikit-learn==0.20.0',
    'matplotlib==2.1.2',
    'PyYAML==3.12',
    'keras==2.2.4',
    'scipy==1.0.0',
    'tensorflow==1.12.0',
    'python-igraph==0.7.1.post6',
    'gem @ git+https://github.com/fantarolf/GEM#egg=GEM'
]


setuptools.setup(
    name="ne4lp",
    version="1.0",
    packages=['ne4lp', 'ne4lp.emb', 'ne4lp.exp'],
    author="Jonah Kresse",
    description="Node Embedding Algorithms for Link Prediction",
    python_requires='>=3.6.0',
    install_requires=PACKAGES,
)
