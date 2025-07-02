# 😷 Face Mask Detection using Convolutional Neural Networks (CNN)

This project implements a convolutional neural network to classify whether a person in an image is wearing a face mask or not. It leverages image-based learning and deep CNN architectures to automate mask detection for safety compliance in public spaces.

---

## Dataset

## 📁 Dataset

The dataset is too large for GitHub.  
You can download it from [this Google Drive link](https://drive.google.com/file/d/1q1-Pduxn0EF1cgFUj4mRxfZSNgyFkUE_/view?usp=sharing).



The dataset consists of images categorized into two classes:

| Class Label | Description                  |
|-------------|------------------------------|
| `mask`      | Person **wearing** a face mask |
| `no_mask`   | Person **not wearing** a face mask |


**Dataset characteristics:**
- Well-labeled, clean face images  
- Balanced across both classes  
- Images resized to 128×128 during preprocessing  
- No missing images or corrupted files  

---

## Libraries Used

- `numpy` – Numerical operations  
- `matplotlib` – Visualization  
- `opencv-python` – Image loading and processing  
- `tensorflow / keras` – Model building and training  
- `scikit-learn` – Evaluation and preprocessing  

---

## Workflow

### 1. Image Preprocessing
- Load image using OpenCV  
- Resize to 128x128  
- Normalize pixel values (divide by 255)  
- Reshape to 4D batch for model input  

### 2. Model Architecture
- Convolutional Neural Network (CNN) built from scratch  
- Two Conv2D + MaxPooling layers  
- Flattened and passed through Dense layers  
- Dropout used for regularization  
- Output layer with softmax activation for 2-class classification  

### 3. Training
- Loss function: `sparse_categorical_crossentropy`  
- Optimizer: `Adam`  
- Epochs: 10  
- Validation split: 10% of training data  
- Accuracy reached ~96% on training and ~92% on validation data  

### 4. Inference
- Accepts custom image input  
- Predicts one of two classes: `mask`, `no_mask`  
- Uses highest probability from softmax output  

---

## Model Performance

**Final Metrics:**
- Training Accuracy: **~96%**  
- Validation Accuracy: **~92%**  
- Model overfit minimal, generalizes decently on similar face images  
- May struggle on poorly lit, angled, or partially obstructed faces  

---

## Example Prediction

**Input Prompt:**
```python
Enter your image path: 'content/xyz/'
```

**Output:**
```python
1
the person is wearing a mask
```


---

## How to Run

### Prerequisites
- Python 3.8 or higher  
- pip package manager  

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/face-mask-detection.git
cd face-mask-detection
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Run the Notebook**
```bash
jupyter notebook Face_Mask_Detection_using_CNN.ipynb
```
Play around and make changes trying different inputs and predictions from the dataset! 

