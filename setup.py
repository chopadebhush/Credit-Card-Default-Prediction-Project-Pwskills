import setuptools

with open("README.MD","r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"


REPO_NAME = "Credit-Card-Default-Prediction-Project-Pwskills"
AUTHOR_USER_NAME = "techfreakydeepak"
SRC_REPO = "ccdp"
AUTHOR_EMAIL = "itsdeepaksingh00@gmail.com"

setuptools.setup(
    name = SRC_REPO,
    version = __version__,
    author = AUTHOR_USER_NAME,
    author_email = AUTHOR_EMAIL,
    description = "A small Python package for Predicting Credit Card Default.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls= {
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"
    },
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src")
)