# CQA_using_PLM

This repository contains the code and resources for the research project titled **"Column Qualifier Annotation Using Pretrained Language Model"**. The project focuses on automating the annotation of table columns, particularly handling n-ary relationships using a BERT-based model. The work builds on Column Type Annotation (CTA), Column Property Annotation (CPA), and Column Qualifier Annotation (CQA) tasks to improve table understanding and annotation through Pretrained Language Models (PLMs).

## Project Overview

The goal of this project is to enhance automated table annotation by developing a robust BERT-based model for handling:
- **Column Type Annotation (CTA)**: Determining the type of data in each column.
- **Column Property Annotation (CPA)**: Understanding the relationships between columns.
- **Column Qualifier Annotation (CQA)**: Adding qualifiers and context to relationships, capturing complex n-ary associations.

### Key Features
- **Automated Table Annotation**: Designed to process and annotate columns based on semantic relationships.
- **BERT-based Model**: The model is built on top of BERT and fine-tuned for handling column annotations in tabular data.
- **Preprocessing Tools**: Advanced preprocessing techniques are included to improve the accuracy and generalization of the model.
- **Multiple Run Configurations**: Five different CQA configurations are supported, each representing different strategies for training and evaluation.
  
### Contributions
- Development of a robust model architecture for table annotation using BERT.
- Innovative preprocessing and data handling techniques for improved performance.
- Comprehensive evaluation framework that assesses performance under multiple configurations.

## Files and Structure

- `model.py`: Implements the BERT-based model architecture for multi-output classification, handling CTA, CPA, and CQA tasks【22†source】.
- `train_multi.py`: Script for training the model with options for running in "sequential" or "round robin" mode. It supports logging, evaluation, and error analysis after each task and epoch【23†source】.
- `util.py`: Contains utility functions for evaluation, such as calculating F1 scores and analyzing confusion matrices【24†source】.
- `pt_dataset.py`: Handles the dataset creation and collation for the model, preparing data for the CTA, CPA, and CQA tasks.
- `Preprocessing`: This folder contains all necessary preprocessing files that handle data augmentation, tokenization, and preparation for training【21†source】.
- `instructions.txt`: Provides details on the different CQA run options and how to configure the training setup【21†source】.

## Preprocessing

The `Preprocessing` folder contains the scripts required to prepare and process the tabular data. These scripts handle tasks such as:
- **Data Augmentation**: Splitting and augmenting tables for better model generalization.
- **Tokenization**: Converting table data into tokenized inputs suitable for BERT-based models.

## Configuration

The project supports five distinct CQA run configurations, each targeting different preprocessing and training strategies:
1. **CQA_run1**: Excludes all tables with nonzero values in subject, object, and qualifier columns and includes one example category in training.
2. **CQA_run2**: Excludes all tables with nonzero values and excludes one example category in both training and validation.
3. **CQA_run3**: Includes all tables with nonzero values by switching subject columns and includes one example category in training.
4. **CQA_run4**: Same as CQA_run3 but excludes one example category in training and validation.
5. **CQA_run5**: Randomly splits tables into two, includes both example categories after the split in training and validation【21†source】.

These configurations can be selected using the `--run_option` argument when training the model.

## Usage

- ### Prerequisites

- Python 3.8+
- PyTorch
- Huggingface's Transformers
- Pandas, Numpy, Scikit-learn, SciPy
- Matplotlib, Seaborn
- WandB for experiment tracking
- Spacy for NLP
- Scikit-learn for evaluation

### Installation

To install the required dependencies, use the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

**requirements.txt**:
```plaintext
_libgcc_mutex==0.1
_openmp_mutex==5.1
appdirs==1.4.4
asttokens==2.0.5
backcall==0.2.0
beautifulsoup4==4.12.2
blas==1.0
blis==0.7.9
bottleneck==1.3.5
brotli==1.0.9
brotli-bin==1.0.9
brotlipy==0.7.0
bs4==0.0.1
ca-certificates==2023.01.10
catalogue==2.0.8
certifi==2022.12.7
cffi==1.15.1
charset-normalizer==2.0.4
commonregex==1.5.4
confection==0.0.4
contourpy==1.0.5
cryptography==39.0.1
cycler==0.11.0
cymem==2.0.7
dbus==1.13.18
decorator==5.1.1
docker-pycreds==0.4.0
elastic-transport==8.4.0
elasticsearch==8.3.3
emoji==2.2.0
en-core-web-sm==3.5.0
executing==0.8.3
expat==2.4.9
fontconfig==2.14.1
fonttools==4.25.0
freetype==2.12.1
ftfy==6.1.1
giflib==5.2.1
gitdb==4.0.10
gitpython==3.1.31
glib==2.69.1
gst-plugins-base==1.14.1
gstreamer==1.14.1
huggingface-hub==0.0.8
icu==58.2
idna==3.4
imageio==2.31.2
importlib_resources==5.2.0
intel-openmp==2021.4.0
ipython==8.12.0
jedi==0.18.1
jinja2==3.1.2
joblib==1.1.1
jpeg==9e
kiwisolver==1.4.4
krb5==1.19.4
langcodes==3.3.0
langdetect==1.0.9
lcms2==2.12
ld_impl_linux-64==2.38
lerc==3.0
libbrotlicommon==1.0.9
libbrotlidec==1.0.9
libbrotlienc==1.0.9
libclang==14.0.6
libclang13==14.0.6
libdeflate==1.17
libedit==3.1.20221030
libevent==2.1.12
libffi==3.4.2
libgcc-ng==11.2.0
libgfortran-ng==11.2.0
libgfortran5==11.2.0
libgomp==11.2.0
libllvm14==14.0.6
libpng==1.6.39
libpq==12.9
libstdcxx-ng==11.2.0
libtiff==4.5.0
libuuid==1.41.5
libwebp==1.2.4
libwebp-base==1.2.4
libxcb==1.15
libxkbcommon==1.0.1
libxml2==2.10.3
libxslt==1.1.37
lz4-c==1.9.4
markupsafe==2.1.2
matplotlib==3.7.1
matplotlib-base==3.7.1
matplotlib-inline==0.1.6
mkl==2021.4.0
mkl-service==2.4.0
mkl_fft==1.3.1
mkl_random==1.2.2
multipledispatch==0.6.0
munkres==1.1.4
murmurhash==1.0.9
ncurses==6.4
nspr==4.33
nss==3.74
numexpr==2.8.4
numpy==1.20.2
openssl==1.1.1t
packaging==23.0
pandas==1.2.4
parso==0.8.3
pathtools==0.1.2
pathy==0.10.1
pcre==8.45
pexpect==4.8.0
pickleshare==0.7.5
pillow==9.4.0
pip==23.0.1
ply==3.11
pooch==1.4.0
preshed==3.0.8
prompt-toolkit==3.0.36
protobuf==4.23.2
psutil==5.9.5
ptyprocess==0.7.0
pure_eval==0.2.2
pybind11==2.10.4
pycparser==2.21
pydantic==1.10.7
pygments==2.11.2
pyopenssl==23.0.0
pyparsing==3.0.9
pyqt==5.15.7
pyqt5-sip==12.11.0
pysocks==1.7.1
python==3.8.16
python-dateutil==2.8.2
pytz==2022.7
pyyaml==6.0
qt-main==5.15.2
qt-webengine==5.15.9
qtwebkit==5.212
readline==8.2
recognizers-text==1.0.2a2
requests==2.28.1
sacremoses==0.0.53
scikit-learn==0.24.1
scipy==1.6.2
sentry-sdk==1.24.0
setproctitle==1.3.2
setuptools==67.7.2
sip==6.6.2
six==1.16.0
smart-open==6.3.0
smmap==5.0.0
soupsieve==2.4.1
spacy==3.5.2
spacy-legacy==3.0.12
spacy-loggers==1.0.4
sqlite==3.41.2
srsly==2.4.6
stack_data==0.2.0
thefuzz==0.19.0
thinc==8.1.9
threadpoolctl==2.2.0
tk==8.6.12
tokenizers==0.10.3
toml==0.10.2
torch==1.8.1
tornado==6.2
traitlets==5.7.1
transformers==4.6.0
typer==0.7.0
typing_extensions==4.5.0
urllib3==1.26.15
wandb==0.15.3
wasabi==1.1.1
wcwidth==0.2.5
wheel==0.40.0
xz==5.2.10
zipp==3.11.0
zlib==1.2.13
zstd==1.5.5
```

### Running the Model
To train the model with a specific configuration, use the following command:
```bash
python train_multi.py --run_mode round_robin --run_option CQA_run1 --epochs 10
```

### Evaluation
Model performance is evaluated using precision, recall, F1 score, and confusion matrices. The evaluation is conducted after each epoch, and results are saved for both training and validation datasets.

## License
This project is licensed under the terms of the Apache 2.0 License.
