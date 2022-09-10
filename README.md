# X-ray Segmentation with detectron2 (image-captioining pretraining)

<br>

**Sep 8, 2022**
* First release of the project.

<br>

<img src="./docs/1.png" align="center"/>
In this project, asd.

## Dependency installation

The code was successfully built and run with these versions:

```
pytorch-gpu 1.10.1
cudatoolkit 11.3.1
opencv 4.6.0.66
scikit-learn 1.0.2
transformers 4.20.1
detectron2 0.6cuda113
numpy 1.22.4

Note: You can also create the environment I've tested with by importing _environment.yml_ to conda.
```


## Preprocessing Data

The image captioning transformer model and CNN both is trained on CANDID-PTX dataset, for this aim you can preprocess the data using dataloader in captioning/data/transformer_dataloader.py (for transformer) and captioning/data/cnn_dataloader.py (for CNN)

The Segmentaion model can be trained on both ChestX-Det dataset that you can download from [here](https://github.com/Deepwise-AILab/ChestX-Det-Dataset) and CANDID-PTX dataset that you have to recieve access to it from [here](https://auckland.figshare.com/articles/dataset/CANDID-PTX/14173982) and CANDID-PTX dataset paper can be found [here](https://pubs.rsna.org/doi/10.1148/ryai.2021210136)
As detectron2 is implemented to train the segmentation model, the data is better to be in coco style, for this aim you can use the dataloader file in segmentation/data/dataloader for both datasets to prepare them in coco style and load for training process.
Also for verification of your annotations in coco style, you can use the visualtization file in segmentation/data/data_visualize.py


You should place the data in the following structure:
```
/segmentation/
  data/
    CANDID_PTX/
        annotations/
        images/

    ChestX_Det/
        annotations/
        images/
```


<br>

## Training

This code is written using [detectron2](https://github.com/facebookresearch/detectron2). You can train the captioning models with running train.py and cnn_train.py in /captioning/ directory for transformer and CNN model respectively. 
<br>
You can also config the model and training parameters in ./configuration.py.

```
...
```

<br>

## Reference 

[//]: # ([ArXiv's paper]&#40;https://arxiv.org/pdf/2008.02063&#41;)
```
@inproceedings{feng2021curation,
  title={Curation of the CANDID-PTX Dataset with Free-Text Reports},
  author={Sijing Feng, Damian Azzollini, Ji Soo Kim, Cheng-Kai Jin, Simon P. Gordon, Jason Yeoh, Eve Kim, Mina Han, Andrew Lee, Aakash Patel, Joy Wu, Martin Urschler, Amy Fong, Cameron Simmers, Gregory P. Tarr, Stuart Barnard, Ben Wilson},
  booktitle={Radiology: Artificial Intelligence},
  year={2021}
}
```


<br><br><br>
