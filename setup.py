from distutils.core import setup

setup(
      name='scg',
      version='0.3.1',
      description='Python tools for inferring clonal genotypes from single cell data.',
      author='Andrew Roth',
      author_email='andrewjlroth@gmail.com',
      url='http://compbio.bccrc.ca',
      package_dir = {'': 'lib'},    
      packages=[ 
                'scg',
                ],
      scripts=['scg']
     )
