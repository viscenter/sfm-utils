import setuptools

# Import README as a long description
with open("README.md", "r") as fh:
    long_desc = fh.read()

# Setup
setuptools.setup(name="PySfMUtils",
                 version="1.0.1",
                 author="Seth Parker",
                 author_email="c.seth.parker@uky.edu",
                 description="A Python library for interacting with Structure-from-Motion projects",
                 long_description=long_desc,
                 long_description_content_type="text/markdown",
                 url="https://github.com/viscenter/sfm-utils",
                 packages=setuptools.find_packages(),
                 classifiers=[
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
                     "Operating System :: OS Independent"
                 ],
                 install_requires=['numpy>=1.15'],
                 python_requires='>=3.6'
                 )
