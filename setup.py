import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open("requirements.txt") as fh:
    required = fh.read().splitlines()

setuptools.setup(
    name="PhraseExtraction",
    packages=['phraseextraction'],
    version="0.0.1",
    author="FMR LLC",
    author_email="phraseextraction@fmr.com",
    description="A simple research library for extracting key phrases from text using different NLP Techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fidelity/PhraseExtraction",
    install_requires=required,
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    keywords='Keyword_extraction Phrase_extraction',
    
    project_urls={
        "Source": "https://github.com/fidelity/PhraseExtraction"
    }
)

