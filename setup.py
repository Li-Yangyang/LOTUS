from setuptools import setup, find_packages
import os, sys, re
import codecs

NAME = "LOTUS"
PACKAGES = find_packages(where='src')
META_PATH = os.path.join("src", "LOTUS", "__init__.py")
EXTRA_REQUIRE = {
    "advanced-interp": ["rbf", "torch", "gpytorch"],
    "doc": [
        "sphinx-book-theme",
    ],
}
CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
]

HERE = os.path.dirname(os.path.realpath(__file__))

def readme():
    with open("README.md") as f:
        return f.read()

with open('requirements.txt') as infd:
    INSTALL_REQUIRES = [x.strip('\n') for x in infd.readlines()]
    print(INSTALL_REQUIRES)

def read(*parts: str) -> str:
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()

def find_meta(meta: str, meta_file: str = read(META_PATH)) -> str:
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), meta_file, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


setup(
    name=NAME,
    use_scm_version={
            "write_to": os.path.join(
                "src", NAME, "{0}_version.py".format(NAME)
            ),
            "write_to_template": '__version__ = "{version}"\n',
        },
    author=find_meta("author"),
    author_email=find_meta("email"),
    maintainer=find_meta("author"),
    url=find_meta("url"),
    license=find_meta("license"),
    description=find_meta("description"),
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=PACKAGES,
    package_dir={"": "src"},
    package_data={'LOTUS': ['package_data/gcoglib/*', 'package_data/ewdiff/*', 'package_data/linelist/*']},
    include_package_data=True,
    python_requires=">=3.7",
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRA_REQUIRE,
    zip_safe=False
)
