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
## 3. Introduction to the important Program File
__main.py:__ The main process file to run our neural machine translation model. In main.py, we executed a five-step process: (1) Load the dataset. (2) Define the model and training details. (3) Print out the configuration of the model and the hyperparameter setting for training. (4) Train the Model (5) Evaluate the best saved model on the test dataset. 

__preprocess.py:__ The program file that defines a set of functions that has been applied to load the data and prepare the batches for neural network.

__NMT_Seq2Seq.py:__ The model file that defines the Sequence to Sequence Model and the beamsearch decoding process.

__train.py:__ This script defines how we apply backpropagration to optimize the network. 

__layers/:__ Under this directory, we have defined the encoder and decoder structure and attention mechanism.

__evaluation/:__ Under this repository we have a BLEU score computation python script from [Google Neural Machine Translation Tutorial](https://github.com/tensorflow/nmt).
## 4. Prepare Dataset
We have a preprocessed IWLST 2015 EN -> VI dataset avaible in this [google drive link](https://drive.google.com/drive/folders/1DvnSJO4sFVspox4e8rWVZdRMb0s40HZb?usp=sharing), which can be downloaded to train the model. 

To prepare your own translation dataset, you will need to prepare 8 files. The source sentences and their translations need to be stored in two separate files with the extension defined as the abbrreviation of source language and target language. For example for a file that contains source sentences in English should be named as "xxx.en" abd for a file that contains target translation sentences in French should be named as "xxx.de". There will be 2 files for training dataset, 2 files for validation dataset, and 2 file for testing dataset. They should be named in the format of "[split].[language abbreviation]". For example if it is a training datafile in English, it should be named as "train.en". In these dataset file, each line of data will be tokenized, which means that every word and punctuation is separated by an empty space. Last, you will also need to define two vocabulary files that contain a list of unique words in the source language and target translation language. Each unique word will take one line. The first three words for both Vocabulary file will be: "\<s\>", "\</s\>" and "\<unk\>", which are useful for neural network to process the sentence. 

If you have any questions with regard to this instruction on how to prepare your own dataset, you can refer to the IWLST 2015 dataset that we provided as an example to prepare your own dataset. 

## 5. How to run the code
Once you have meet with all the prerequisite and have the dataset prepared, you can start to run our code. The first step is to reconstruct the software environment for the code. We provide a Conda virtual environment file to help users to reconstruct the same software environment as the one we used to run the code. Using the following command to create the same software environment:
```
conda env create -f environment.yml
```
Then a virtual environment named "pytorch-py3" will be created. To lauch this environment, simply run:
```
source activate pytorch-py3
```
Once you have launched the virtual environment, you can start to run our code. To train our model, you will need to run the file "main.py" with the following command:
```
python main.py --data_path path/to/data --trained_model_path /path/to/save/model --sr en --tg de
```
You need to define at least four things in order to run this code: the directory for the dataset, the directory to save the trained model, the source language, and the target language. You can also set other parameters, for example batch size by using the "--batch_size" flag. For a list of parameters that you can set, please refer to the beginning of the main.py
