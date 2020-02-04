import setuptools

requirements = []
with open("requirements.txt", "r") as f:
    for line in f.readlines():
        line = line.replace("\n", "")
        requirements.append(line)


setuptools.setup(
    name='cleanup',
    version='0.0.1',
    description='Image cleaning.',
    author='Leonardo Angelo',
    author_email='leonardoangelo8@gmail.com',
    packages=setuptools.find_packages(),
    setup_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
