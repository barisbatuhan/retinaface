# RetinaFace: Single-shot Multi-level Face Localisation in the Wild

## Description

This project is an unofficial implementation of the paper "RetinaFace: Single-shot Multi-level Face Localisation in the Wild" in Julia programming language. 

**Abstract of the Paper:** Though tremendous strides have been made in uncontrolled face detection, accurate and efficient 2D face alignment and 3D face reconstruction in-the-wild remain an open challenge. In this paper, we present a novel single-shot, multi-level face localisation method, named RetinaFace, which unifies face box prediction, 2D facial landmark localisation and 3D vertices regression under one common target: point regression on the image plane. To fill the data gap, we manually annotated five facial landmarks on the WIDER FACE dataset and employed a semi-automatic annotation pipeline to generate 3D vertices for face images from the WIDER FACE, AFLW and FDDB datasets. Based on extra annotations, we propose a mutually beneficial regression target for 3D face reconstruction, that is predicting 3D vertices projected on the image plane constrained by a common 3D topology. The proposed 3D face reconstruction branch can be easily incorporated, without any optimisation difficulty, in parallel with the existing box and 2D landmark regression branches during joint training. Extensive experimental results show that RetinaFace can simultaneously achieve stable face detection, accurate 2D face alignment and robust 3D face reconstruction while being efficient through single-shot inference.

## Sample Result

![Sample Image Detection Result](./data/results/evaluated.jpg)

## Requirements

* Julia >= v1.5.3 (Latest is preferred)

* Knet >= v1.4.5 (Latest is preferred)

* CUDA >= 11.0 (Optional: Only required for running the code in GPU, latest is preferred)

* Python v3.x (Optional: Only required if WIDER FACE validation data evaluation will be made)

## First Setup

* All the packages required to be installed are found under `setup.jl`. To install packages to you can directly run that source file (you can also type `make setup` to run it).

* To train or predict boxes and landmarks by using this repository, please initialize the parameters in `configs.jl` first. Especially please assign correct paths for the dataset directory.

* All the training, predicting and evaluating commands are available in the makefile. Please check that file before trying to run the code.

**Note:** In order to use the `make` command, the operating system should be a linux based distro or macOS. To use this command on Windows 10, please install [**GNU Make**](https://www.gnu.org/software/make/).

## Links

* [WIDERFACE](http://shuoyang1213.me/WIDERFACE/)

* [WIDERFACE Landmark Annotations](https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0)

* [AFLW](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/) (not used in this project but used in the official paper)

* [FDDB](http://vis-www.cs.umass.edu/fddb/) (not used in this project but used in the official paper)

* [ImageNet Weights for ResNet50](https://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat)

* [Weights to Pretrained Models](https://drive.google.com/drive/folders/1GTyTgfmAG2BXvbDDy5n9Jv2ajv1IvWaw?usp=sharing)

**Note:** You can find the default parameters of each of the pretrained models in [**this file**](./weights/info.txt).

## Demonstration of the Network

![Network Graph](./data/readme/network.JPG)

## Progress So Far

* The entire training, predicting and AP evaluation pipelines are implemented. 

* An already pretrained model is created by loading the weights from the [**PyTorch implementation of this paper**](https://github.com/biubug6/Pytorch_Retinaface). 

* Cascaded structure of the model is also included to the model. The same context modules are used but the multitask heads are different for each cascaded structure.

* The model supports both implementations with 3 and 5 lateral connections. However, the only backbone available currently is ResNet50.

## Currently Working On

* Weights available in GitHub with 5 lateral connections are retrieved from intermediate results. A fully training process is running currently. 

## Results & Evaluation

Model | WIDER Easy AP | WIDER Medium AP | WIDER Hard AP |
--- | --- | --- | --- |
Official Paper | 96.6 | 95.9 | 91.1 |
[Official Shared Sub-Model](https://github.com/deepinsight/insightface/tree/master/detection/RetinaFace) | 94.9 | 93.9 | 88.3 |
[PyTorch Weights Loaded](https://github.com/biubug6/Pytorch_Retinaface) | 77.8 | 80.3 | 73.8 | 
*Cascaded_Model_No_Augment* | 70.3 | 75.7 | 65.4 |
*Cascaded_Model_With_Augment* | 83.0 | 84.3 | 73.5 |

Here, AP is calculated by taking the IOU threshold as 0.5.

**Note:** If you just try to predict faces in images, please use the PyTorch weight-transferred model weights, since it is the most stable version right now. Also, Cascaded_Model_With_Augment does not perform well in landmark localization task. Please select not augmented version for landmark localization task if you want to use the full model structure.

## Extra Visualization

![Image2](./data/results/evaluated2.png)

![Image3](./data/results/evaluated3.png)

## What To Do Next

* Support other datasets (FDDB and AFLW) in addition to the WIDERFACE dataset.

* Implement the other evaluation metrics: Area Under Curve (AUC), Failure Rate and Normalized Mean Error (NME).

* Instead of the normal Convolutional Layers in Context Modules, implement [**Deformable Convolutional Layers**](https://arxiv.org/abs/1703.06211) and retrain the model with this structure.

* Add different backbones for shorter processing times, such as MobileNet.

* Add multiple GPU support.

