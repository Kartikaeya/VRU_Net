# VRU_Net
Codebase containing the scripts of training/testing of the proposed VRU-Net for Human Pose Estimation. More details about the project can be found in [this](https://docs.google.com/document/d/1WAAjlRcaQkftHq1hMvORNOwW4jbvuTLPxICMzBCF44Q/edit?usp=sharing) report.

### Setup
#### COCO Dataset
  1. Download the [COCO](https://cocodataset.org/#download) training data from [here](http://images.cocodataset.org/zips/train2017.zip) and unzip it in `$Root/Data/COCO/`. 
  2. Download the [COCO](https://cocodataset.org/#download) validation data from [here](http://images.cocodataset.org/zips/val2017.zip) and unzip it in `$Root/Data/COCO/`.
  3. Download the annotations file from [here](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) and unzip it in `$Root/Data/COCO/annotations/`.
  4. Install the packages written in `$Root/required packages.txt`.
### Directory Structure
After doing the above setup the directory should look like:
```bash
$Root
├── Data
│   ├── COCO
│   │   ├── Annotations
|   |   |   ├── person_keypoints_train2017.json
|   |   |   └── person_keypoints_val2017.json
|   |   ├── train2017
|   |   |   ├── img1.jpg
|   |   |   ├── img2.jpg
|   |   |   └── ...
|   |   └── val2017
|   |       ├── img1.jpg
|   |       ├── img2.jpg
|   |       └── ...
|   └── VRU
│       ├── Annotations
|       |   ├── bbox.json
|       |   └── vru_keypoints_val.json
|       └── images
|           └── val
|               ├── img1.jpg
|               ├── img2.jpg
|               └── ...
├── runs/
├── trained_models/
├── Data_Loader.ipynb
├── Data_Loader.py
├── Inference.ipynb
├── Modify_annotations.ipynb
├── Modify_annotations.py
├── model.ipynb
├── model.py
├── required_packages.txt
├── train.ipynb
└── .gitignore
```
### Training
   1. Make sure that `setup` is done and the files are placed as in `Directory Structure`.
   2. Modify the hyperparameters in the second block of `train.ipynb`.
   3. Run the Notebook `train.ipynb`
