from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='lddl',
    version='0.1.0',
    description=
    'Language Datasets and Data Loaders for NVIDIA Deep Learning Examples',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='github.com/NVIDIA/DeepLearningExamples/tools/lddl',
    author='Shang Wang',
    author_email='shangw@nvidia.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3 :: Only',
    ],
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'dask[complete]==2021.7.1',
        'distributed==2021.7.1',
        'dask-mpi==2021.11.0',
        'pyarrow==4.0.1',
        'mpi4py==3.1.3',
        'transformers==4.16.2',
        'wikiextractor==3.0.6',
        'news-please==1.5.21',
        'cchardet==2.1.7',
        'awscli==1.22.55',
        'wikiextractor @ git+https://github.com/attardi/wikiextractor.git',
    ],
    entry_points={
        'console_scripts': [
            'download_wikipedia=lddl.download.wikipedia:console_script',
            'download_books=lddl.download.books:console_script',
            'download_common_crawl=lddl.download.common_crawl:console_script',
            'preprocess_bert_pretrain=lddl.dask.bert.pretrain:console_script',
            'balance_dask_output=lddl.dask.load_balance:console_script',
            'generate_num_samples_cache=lddl.dask.load_balance:generate_num_samples_cache',
        ],
    },
)
