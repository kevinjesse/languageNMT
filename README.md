## Implementation of Sequence to Sequence Neural Machine Translation Model with Attention Mechanism
## 1. Overview
This code repository contains the implementation of the sequence to sequence neural machine translation model with attention mechanism introduced in the following paper:
[Neural Machine Translation By Jointly Learning to Align and Translate, by Dzmitry Bahdanau, KyungHyun Cho, and Yoshua Bengio](https://arxiv.org/pdf/1409.0473.pdf)
## 2. Prerequisite
The code is successfully tested oin Ubuntu 16.04 with NVIDIA GPU GTX 1080 Ti. In order to run the code with this GPU you need to have the following softwares and libraries installed:
1. Python 3.6
2. [CUDA 8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive)
3. CUDNN 7.1.4
4. [Conda](https://conda.io/miniconda.html)
While we havn't tested this code with other OS systems, we expect it can be runned on any Linux Based OS with a minor adjustment. 

## 3. How to run the code
Once you have meet with all the preresuisites, you can start to run our code. The first step is to reconstruct the software environment for the code. We provide a Conda virtual environment file to help users to reconstruct the same software environment as the one we used to run the code. Using the following command to create the same software environment:
```
conda env create -f environment.yml
```
Then a virtual environment named "pytorch-py3" will be created. To lauch this environment, simply run:
```
source activate pytorch-py3
```
Once you have launched the virtual environment, you can start to run our code. To train a VAG-NMT, you will need to run the file "main.py" with the following command:
```
  python main.py --data_path path/to/data --trained_model_path /path/to/save/model --sr en --tg de
```
You need to define at least four things in order to run this code: the directory for the dataset, the directory to save the trained model, the source language, and the target language. The languages that our model can work with include: English=> "en", German->"de" and French->“fr”.

We have the Preprocessed Multi30K Dataset available in this [link](https://drive.google.com/drive/folders/1G645SexvhMsLPJhPAPBjc4FnNF7v3N6w?usp=sharing), which can be downloaded to train the model.

### Things TODO before Submission
1. Introduce the Data Preprocessing Steps
2. Introduce the Code files in more details
3. Add more comments to the code so that the code can be easier understaood
4. Include the results evaluated on the dataset here.