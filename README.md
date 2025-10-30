# 🎸 Guitar Detection (YOLOv8)

This repository contains a custom YOLOv8 model for detecting guitars in images.  
It builds upon the public [Guitar Detection Dataset](https://www.kaggle.com/datasets/sagarnildass/guitar-detection-dataset) (CC0 Public Domain) by **Sagar Nil Das**.

## 🚀 Overview
- **Model:** YOLOv8 nano (Ultralytics)
- **Dataset:** 70 train / 18 validation / 310 test images
- **Additions:** 25 manually selected negative images to improve background detection
- **Experiments:** 7 training runs (fine-tuning and negative sample tests)
- **Best Run:** `guitar_kaggle_finetune3`  
  Precision ≈ 1.00 Recall ≈ 0.81 Accuracy ≈ 0.83

## 🧩 Features
- Full YOLOv8 training and validation pipeline  
- Custom confusion-matrix generator  
- Negative-sample labeling script  
- Support for evaluating single images

## 📂 Project Structure
```plaintext
KaggleGuitar/
├── src/ 
│ ├── ConfusionMatrixGenerator.py
│ ├── LabelNegatives.py
│ └── Test.py
├── models/ 
│ └── yolov8n.pt
├── data/ 
├── runs/ 
├── requirements.txt
├── .gitignore
└── LICENSE
```
## 🧠 Dataset Attribution
Data from the [Guitar Detection Dataset](https://www.kaggle.com/datasets/sagarnildass/guitar-detection-dataset)  
by **Sagar Nil Das**, released under **CC0: Public Domain**.  
Dataset reorganized and expanded with additional negative samples.

## ⚙️ Installation
git clone https://github.com/jkilshaw/guitar-detection-yolov8.git
cd guitar-detection-yolov8
pip install -r requirements.txt

## 🧪 Usage
### Train
yolo detect train data=data/data.yaml model=models/yolov8n.pt epochs=50 imgsz=640

### Evaluate
python src/ConfusionMatrixGenerator.py

### Detect a single image
python src/Test.py --image path/to/image.jpg

## 🪪 License
- All code in this repository is released under the MIT License.
- Dataset content follows its original CC0 Public Domain license.
- Author: Jack Kilshaw
- GitHub: @jkilshaw
