# Book-Condition-Evaluation-using-YOLOv5
This repository contains a deep learning project focused on evaluating the condition of books by applying object detection using YOLOv5. Developed in Google Colab, the project leverages a custom dataset to train the model, enabling classification of books into various conditions, such as new, used, or damaged.

## Project Overview
The goal of this project is to automate the evaluation of book conditions, providing a scalable and efficient solution for categorizing books based on their visual condition. YOLOv5, a state-of-the-art object detection model, is fine-tuned for this task to achieve optimal performance.

## Key Features
- **Object Detection with YOLOv5**: Fast and accurate object detection for evaluating book conditions.
- **Custom Dataset**: Labeled images representing various conditions (new, used, damaged).
- **Google Colab Integration**: Developed and executed on Google Colab for easy access to resources.

## Dataset
The dataset used for this project is hosted externally and can be accessed [https://universe.roboflow.com/atcom21-gmail-com/damaged-books/dataset/1]. It contains labeled images representing various book conditions. Detailed preprocessing and augmentation were applied to prepare the dataset for model training.

## Model Architecture
The project is built on top of YOLOv5, which runs using PyTorch. The model was trained for 30 epochs with a batch size of 16 on 415x415 resolution images. The fine-tuning was performed using a pre-trained YOLOv5s model, followed by custom weight updates to detect book conditions.

### Workflow
![WhatsApp Image 2024-06-30 at 18 52 44_a0d72880](https://github.com/user-attachments/assets/df0d40a7-c9c9-4a90-b8c0-fed47a0b7fdc)

## Results
### Quantitative Analysis
The YOLOv5 model's performance was assessed using mean average precision (mAP) at various Intersection over Union (IoU) thresholds:
- **Overall Performance**:
  - mAP@.5: 75.9%
  - mAP@.5:.95: 57.5%
  - These results highlight the model's ability to accurately detect both prominent and subtle book damages.
- **Class-Wise Performance**:
  - Ripped Books: mAP@.5 of 46.7%, reflecting good detection of clear tears.
  - Wornout Books: mAP@.5 of 37.7%, showing challenges with less distinct damage patterns.
### Qualitative Analysis
Visual results illustrate the model's detection capabilities:
- Detection of Rips: Effective at identifying both large and small rips.
- Detection of Wear: Improved identification of minor abrasions, though less precise compared to ripped books.
### Figures:
1. **Class-wise average precision for book damages detected by YOLOv5**
<img src="https://github.com/user-attachments/assets/ea671900-994e-4b74-8217-385bcb65478d" width="400" style="display: inline-block;"/>


2. **Detection Results**
<img src="https://github.com/user-attachments/assets/7cfb87c3-5b6d-4d82-9e4a-aec2364da579" width="400" style="display: inline-block;"/>


3. **Precision-Recall Curve**
<img src="https://github.com/user-attachments/assets/caf68a72-e8c3-4b8f-8353-628ef624099b" width="400" style="display: inline-block;"/>


4. **Confusion Matrix**
<img src="https://github.com/user-attachments/assets/89bc3380-63cd-4664-bc07-a28c8777fe6e" width="400" style="display: inline-block;"/>

## Usage
To run this project in Google Colab, follow these steps:
- Clone this repository into your Colab environment.
- Download the dataset from [https://universe.roboflow.com/atcom21-gmail-com/damaged-books/dataset/1] and place it in the appropriate folder.
- Open the BookConditionEvaluation.ipynb notebook.
- Execute the notebook cells to train the model and detect book conditions.

## Sample Commands
**Training**: The model was trained using the following command:
```python
!python train.py --img 415 --batch 16 --epochs 30 --data data.yaml --weights yolov5s.pt --cache
```
**Detection**: To run detection on a sample image:
```python
!python detect.py --source /path/to/image --weights /path/to/weights --conf 0.25
```

## Conclusion
This project successfully demonstrates how deep learning can be used to evaluate book conditions through object detection. YOLOv5's performance on this task highlights its versatility in various classification scenarios.
