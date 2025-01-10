# _Object recognition and computer vision_ project

This repository contains my work on a project for the [Object recognition and computer vision](https://gulvarol.github.io/teaching/recvis24/) class.

It is based on [Test-Time Training with Masked Autoencoders](https://arxiv.org/abs/2209.07522) by Yossi Gandelsman, Yu Sun, Xinlei Chen and Alexei A. Efros.

The code is taken from [this GitHub repository](https://github.com/yossigandelsman/test_time_training_mae), with various adaptations. Note that we removed the files that are not directly useful for Test-Time Training, such as the ones used to to train the feature extractor and the main-task head.

## Installation

To install all required dependencies, you can run:
```bash
conda env create -f environment.yml
conda activate ttt
```

The ImageNet-C dataset can be downloaded from [here](https://zenodo.org/records/2235448#.Yz9OHezMKFw), for instance using `wget` and `tar -xvf`.

For dependecies, datasets, models, and more information, please visit the previously quoted GitHub repository.

## Modifications

Here are some of the modifications I made :
- In `data/tt_image_folder.py`, lines 8-54, I created a class to handle a dataset containing images with several severities or several types of corruption.
- In `main_test_time_training.py`, lines 104-109, I added arguments to perform online TTT; lines 187-216 use these arguments to create the dataset.
- In `engine_test_time.py`, lines 124-126, the model is reinitialized depending on the arguments; lines 177 had to be modified due to the lower number of images that we used.

These changes were mostly sufficient to run the experiments presented in the report.

## Scripts

The folder `scripts` contains some examples of how to launch experiments (baseline, simple TTT MAE, basic online TTT MAE, online TTT MAE with several severities).
