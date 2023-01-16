from setuptools import setup

with open("README.rst") as f:
    readme = f.read()

kwargs = {
    "name": "tictactoe-gpt-finetuning",
    "version": "0.1.0",
    "description": "Python tic tac toe state generator and GPT fine tuning",
    "author": "Nicky Pochinkov",
    "author_email": "work@nicky.pro",
    "url": "https://github.com/pesvut/tictactoe-gpt-finetuning",
    "license": "MIT",
    "keywords": ["tictactoe", "llm", "language-models"],
    "install_requires": ["numpy", "torch", "transformers"],
    "packages": ["tictactoe_gpt_finetuning"],
    "long_description": readme,
}

setup(**kwargs)
