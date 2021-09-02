# Master thesis

[![PDF](https://img.shields.io/badge/PDF-latest-blue.svg?style=flat)](https://github.com/pabloac31/TFM/report.pdf)

This repository contains the code and the report for the master thesis:

<p align="center" style="font-size:larger;">
<i>Predicting venous thromboembolic events in patients with cancer using a new machine learning paradigm</i>
</p>

written by Pablo Álvarez, under the supervision of Oriol Pujol Vila (UB) and José Manuel Soria (Hospital de Sant Pau), submitted to the Facultat de Matemàtiques i Informàtica of the Universitat de Barcelona.

## Abstract
The rise of machine learning in the last decade has facilitated great advances in fields such as medicine, where very powerful models have been developed, capable of predicting certain medical conditions with an accuracy never seen before.

The present work is focused on predicting one of the leading causes of death among patients with cancer: venous thromboembolic events (VTE). Over the years, several statistical models based on clinical/genetic data have been developed, and have made it possible to create some risk assessment tools, like the [Khorana score](https://pubmed.ncbi.nlm.nih.gov/18216292/). However, none of them are based on machine learning. 

In this way, we propose a new model that uses advanced machine learning techniques and is able to outperform all models currently available. Furthermore, the model is based on a very recent and promising learning paradigm that has barely been tested, hence it is a great opportunity for us to explore and evaluate it. 

This breakthrough ultimately has an impact on the patient's quality of life, improving the ability to detect patients at high risk of developing a VTE, who would benefit from preventive treatment.

## Work outline

### Data
- The notebook ```data_preprocessing.ipynb``` is used to create the dataset (stored in  ```data/data_TiC_Onco.csv```). It also includes a simple analysis of the data.

### Learning Using Statistical Invariants ([LUSI](https://link.springer.com/article/10.1007/s10994-018-5742-0))
This work explores a new learning paradigm based on statistical invariants that act as a teacher during learning.
- The notebook ```SVM_I.ipynb``` was designed for the experiments performed with the LUSI approach, in order to test the SVM&I algorithm implemented in ```lusi.py```.

### Results validation
- The notebook ```validating_results.ipynb``` contains the validation of the results reported in the [reference paper](https://pubmed.ncbi.nlm.nih.gov/29588512/) for the Khorana and TiC-Onco risk scores, using our own methodology.

### Improving the TiC-Onco risk score
- The notebook ```improving_TiC_Onco_score.ipynb``` collects all the experiments performed with machine learning models (including SVM_I), with all the results obtained to be reviewed if necessary.
