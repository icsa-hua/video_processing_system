import setuptools 

with open("README.md", "r") as file: 
    long_description = file.read()

with open("requirements.txt", "r") as file:
    requirements = file.read().splitlines()

setuptools.setup(
    name="VIDPS",
    version="0.1.0",
    description="A package for a video processing system", 
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="", 
    packages=setuptools.find_packages(include=['obs_system', 'obs_system.*']),
    install_requires=requirements,
    python_requires='>=3.9'
)