# DATA620004-FinalPJ
Final project for DATA620004:Neural Network and Deep learning, Spring 2022.

# **Assignment 1**
## **A Visualized Video of the Semantic Segmentation Model**

### **Model**

Open-sourced model could be found at [model source](https://github.com/VainF/DeepLabV3Plus-Pytorch). 

### **Visualization**

The driving video is downloaded from [video source](https://www.cityscapes-dataset.com/file-handling/?packageID=12).Then put the frame-to-frame figures of the video in `image_file_path` and run the following command, which would put the predicted figures in `test_results`.

```shell
python predict.py --input image_set_file_path  --dataset cityscapes --model deeplabv3plus_mobilenet --ckpt checkpoints/assignment1_deeplabv3plus_mobilenet_cityscapes.pth --save_val_results_to test_results
```

Then run the following command to concat the pictures and generate the video.

```python
python demo_video.py
```

# **Assignment 2**

## **Faster R-CNN With Differently Initialized Backbone**
This is a pytorch implementation of Faster R-CNN following the work of https://github.com/bubbliiiing/faster-rcnn-pytorch with some adaptations. The dataset used is PASCAL VOC 2007.

### **Requirements**
My experiment environment:
>torch == 1.11.0 CUDA Version: 11.3

### **Training**
1. Using VOC format for training, you need to download the VOC07 data set, decompress it and put it in the root directory.
2. Modify `annotation_mode = 2` in `voc_annotation.py`, then
```bash
python voc_annotation.py
```
generate `2007_train.txt` and `2007_val.txt` in the root directory.
3. Using different backbone initialization methods:
* Initialized randomly: Modify `pretrained = False` in `train.py`.
* Initialized with ImageNet trained backbone parameters: Modify `pretrained = True` in `train.py`, and modify `maskRCNNBackbone = False` in `nets/resnet50.py`. The parameters `resnet50-19c8e357.pth` will be automatically downloaded to the `model_data` path. 
* Initialized with backbone parameters of COCO trained Mask R-CNN: Modify `pretrained = True` in `train.py`, and modify `maskRCNNBackbone = True` in `nets/resnet50.py`. The parameters `maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth` will be automatically downloaded to the `model_data` path. 
4. The default parameters of `train.py` are used to the VOC dataset, and start training by
```bash
python train.py
```
5. The plotted curves and the trained model parameters will be stored in the logs folder.

### **Predict**
1.Prediction requires two files, `frcnn.py` and `predict.py`. Modify the parameter `model_path` when initializing the model in `predict.py` to refer to the trained weights file.

2. 
```bash
python predcit.py
```
After running, enter the image path to detect.

### **Evaluation**
1. Modify the parameter *model_path* when *get_map.py* initializes the model to refer to the trained weights file.
 
2. To get the evaluation results (saved in the *map_out* folder), run
```bash
python get_map.py
```

# **Assignment 3**

Using Transformer for CIFAR100 image classification task.

### **Requirements**
This is my experiment environment:
- python 3.7.11
- pytorch 1.10.2+cu113

### **Usage**
`model.py` defines the functions to build the Transformer model. `utils.py` defines the functions for processing data, data augmentation and configuring hyperparameters.
#### **1. enter directory**
```bash
cd Assignment3
```

#### **2. run tensorboard(optional)**
Install tensorboard
```bash
pip install tensorboard
mkdir runs
tensorboard --logdir='runs' --port=6006 -- host='localhost'
```
the log file of  tensorboard can be found at `./Assignment3/runs`.

#### **3. train the model**
For two different experiments, choose one `.py` file to train.
```bash
# train transformer
python train_200.py 
python train_500.py 
```
and after training, the model will be save at `checkpoint_200`, `checkpoint_500` respectively.


#### **4. test the model**
Test the model using `test.py`
```bash
python test.py 
```

