from setuptools import find_packages, setup

setup(
    name='frugalCopyla',
    version='0.0.2',
    description='tbc',
    author='Dan Manela',
    author_email='tbc',
    install_requires=[
        'jax==0.4.7',
        'jaxlib==0.4.6',
        'jupyter-server==1.23.0',
        'jupyter_client==7.4.4',
        'jupyter_core==4.11.2',
        'jupyterlab==3.5.0',
        'jupyterlab-pygments==0.2.2',
        'jupyterlab_server==2.16.2',
        'matplotlib==3.6.2',
        'matplotlib-inline==0.1.6',
        'numpy==1.23.4',
        'numpyro==0.11.0',
        'pandas==1.5.3',
	'patsy==0.5.3',
        'optax==0.1.4',
        'tensorflow_probability == 0.18.0'
    ],
    packages=find_packages(
        where='frugalCopyla',
        include=['model', 'copula_lpdfs']
    )
)
