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



