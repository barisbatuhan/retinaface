# RetinaFace: Single-shot Multi-level Face Localisation in the Wild

* **Author:** Barış Batuhan Topal
* **Contact:** barisbatuhantopal@gmail.com / baristopal20@ku.edu.tr

## Description

This project is an unofficial implementation of the paper "RetinaFace: Single-shot Multi-level Face Localisation in the Wild" in Julia programming language. 

**Abstract of the Paper:** Though tremendous strides have been made in uncontrolled face detection, accurate and efficient 2D face alignment and 3D face reconstruction in-the-wild remain an open challenge. In this paper, we present a novel single-shot, multi-level face localisation method, named RetinaFace, which unifies face box prediction, 2D facial landmark localisation and 3D vertices regression under one common target: point regression on the image plane. To fill the data gap, we manually annotated five facial landmarks on the WIDER FACE dataset and employed a semi-automatic annotation pipeline to generate 3D vertices for face images from the WIDER FACE, AFLW and FDDB datasets. Based on extra annotations, we propose a mutually beneficial regression target for 3D face reconstruction, that is predicting 3D vertices projected on the image plane constrained by a common 3D topology. The proposed 3D face reconstruction branch can be easily incorporated, without any optimisation difficulty, in parallel with the existing box and 2D landmark regression branches during joint training. Extensive experimental results show that RetinaFace can simultaneously achieve stable face detection, accurate 2D face alignment and robust 3D face reconstruction while being efficient through single-shot inference.

## Useful Links

* [**Paper Link**](https://openaccess.thecvf.com/content_CVPR_2020/papers/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.pdf)

* [**Research Log**](https://docs.google.com/document/d/1fF8Y2ZG3iQvLiHqBY47O8yGQFobWY9JDyNRvDlUqJPQ/edit?usp=sharing) 

* [**Final Presentation**](https://docs.google.com/presentation/d/1lBw68_IdbSe_0n2KAlupRnDulvfzNrUMwx3sBkNl9p8/edit?usp=sharing)

* [**Data Sheet**](https://docs.google.com/spreadsheets/d/1Si1-91wCge3aq7liSTSFxGuJb3fO_-xHlAIzQkaLEyU/edit?usp=sharing) 

* [**Tech Report**](https://www.overleaf.com/read/pbtyskcsdgyt)

## First Setup

* All the packages required to be installed are found under `setup.jl`. To install packages to a virtual environment you can directly run that source file (you can also type `make setup` to run it), or you can install the packages in that file manually to run this project on the actual device.

* All of the implementation steps are summarized with their codes in the `progress.ipynb` notebook and `main.jl` file. According to your preferences, please check these files before give a deeper look on this repository. For running `main.jl` file, a **makefile** is also included to the repository. By using the commands there, you can either run the project with a virtual environment by typing `make run-virtual` or you can directly run by entering `make run` command.

**Note:** In order to use the `make` command, the operating system should be a linux based distro or macOS. To use this command on Windows 10, please install [**GNU Make**](https://www.gnu.org/software/make/).

## Progress So Far

* All image readings and augmentation processes are implemented for WIDERFACE dataset. Please check `BBTNet/utils/*.jl` and `BBTNet/datasets/WIDERFACE.jl` for the source code.

* Dense and Convolutional Layers are implemented under `BBTNet/layers/core.jl` and a custom layer structure called **Conv2D_Block** is implemented under `BBTNet/layers/conv2d_block.jl`. Conv2D_Block struct stacks multiple convolutional layers that have different activation functions, kernel sizes, filter sizes, etc.. It will be useful in next steps.

* A notebook called `progress.ipynb` and a source file named `main.jl` are created for summarizing the completed steps and providing a short guidance for the usage of the commands.

* A makefile is created for running the project and setting up the repository.

## What To Do Next

* ResNet50 backbone must be implemented.

* Upsampling part of the Feature Pyramid Network should be added.

* A loss function and decision mechanisms for bounding boxes must be implemented.

* ...

