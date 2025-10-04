

# Environment Configuration
* Python3.6/3.7/3.8
* Pytorch
* numpy
* pandas
* matplotlib
* Ubuntu/Windows
* It is best to use GPU training
* 
# Dataset download address：
* The worm detection dataset can be obtained in worm data(https://drive.google.com/drive/folders/1PM4Rvrz-V6p-xqAEWsz66tAKu4W5x8Mc), which can extract and recover training.
* Part of the available data is in the Datasets folder
* The video used can be found in train/images/


## Training methods
* Ensure that the dataset is prepared in advance
* To train using a single GPU or CPU:
`
detect train data=datasets/wheat/my data.yaml model=ultralytics/cfg/models/v8/yolov8s CBAM.yaml pretrained=False epochs=300 batch=16 lr0=0.01 resume=True #Need to modify according to one's own actual situation 
`


* If you want to specify which GPU devices to use, you can add 'CUDA_VISIBLEDEVICES=0.3' before the instruction (for example, I only need to use the first and fourth GPU devices in the device)
* `CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 train_multi_GPU.py`

# Precautions
*When using training scripts, be sure to set '-- data path' to the root directory where you store the 'DRIVE' folder**
*When using prediction scripts, set 'weights_path' to your own generated weight path.

# Load a model
`
from ultralytics import YOLO
model = YOLO("path/to/best.pt")  # load a  model
`
# Validate the model
`
metrics = model.val()  # no arguments needed, dataset and settings remembered
`
