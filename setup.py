import os
from setuptools import setup,find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "CreditModels",
    version = "0.0.1",
    author = "Nicola Jean",
    author_email = "nicola.jean@gmail.com",
    description = """Analytical Credit Risk Models""",
    license = "",
    url = "http://localhost:8002",
    packages = ['win_executables'] + find_packages(),
    zip_safe = False,
    package_data = {'win_executables': ['*.bat']},
    long_description=read('README'),
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Topic :: Utilities",
    ],
)
