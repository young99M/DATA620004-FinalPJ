# DATA620004-FinalPJ
Final project for DATA620004, Spring 2022.

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

