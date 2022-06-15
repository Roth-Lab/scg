from setuptools import find_packages, setup

setup(
      name="scg",
      version="0.4.0",
      description="Python tools for inferring clonal genotypes from single cell data.",
      author="Andrew Roth",
      author_email="andrewjlroth@gmail.com",
      url="http://compbio.bccrc.ca",
      package_dir={"": "lib"},
      packages=find_packages(where="lib"),
      entry_points={
        'console_scripts': [
            'scg= scg.cli:main',
        ]
    }
)
