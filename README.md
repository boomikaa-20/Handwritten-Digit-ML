# Handwritten Digit Recogniser

An ML project that recognises handwritten digits from 0 to 9 using the MNIST dataset, PCA, and Support Vector Machine (SVM). The project includes a standalone recognition program and an interactive web application.

## Project Structure

```text
Handwritten-Digit-ML/
├── basic-code/
│   ├── digit_recognizer.py
│   ├── digit_model.joblib
│   ├── 2.png
│   ├── 3.png
│   └── what_the_model_sees.png
│
├── web-app/
│   ├── app.py
│   ├── digit_recognizer.py
│   ├── train_model.py
│   ├── requirements.txt
│   └── models/
│       ├── digit_model.joblib
│       └── metrics.json
│
├── ML-Abstract.pdf
├── .gitignore
└── README.md
```

## 1. Basic Code

The `basic-code/` folder contains the standalone digit recognition program, its trained model, and sample images.

**Features**
- Recognises handwritten digits from 0 to 9.
- Uses PCA for dimensionality reduction.
- Uses an SVM classifier with an RBF kernel.
- Includes sample images and a model preview.

### Run the basic program

Open a terminal in the `basic-code` folder and install its required packages:

```bash
python -m pip install scikit-learn numpy pillow joblib
```

Train the model if needed:

```bash
python digit_recognizer.py train --fast
```

To predict a sample image, use:

```bash
python digit_recognizer.py predict my_digit.png
```

Replace `my_digit.png` with the path to your own digit image.

## 2. Web Application

The `web-app/` folder contains the interactive Streamlit application and the files required for model training and prediction.

**Features**
- Drawing canvas for handwritten digits.
- Digit prediction using the trained ML model.
- Image preprocessing and a processed-image preview.
- Model evaluation and prediction results.

### Run the web application

From the `web-app` folder, create and activate a virtual environment if needed, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the application:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

If the trained model files are missing, train the models first:

```bash
python train_model.py
```

## Technologies Used

- Python
- NumPy
- Scikit-learn
- PCA
- Support Vector Machine (SVM)
- Streamlit
- Pillow
- Joblib
- MNIST dataset

## Project Goal

The goal is to demonstrate handwritten digit recognition using machine learning and compare a standalone implementation with an interactive web application.