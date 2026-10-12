# Handwritten Digit Recogniser

Draw a digit (0-9) on a web canvas and a PCA + SVM model trained on MNIST recognises it.

## Files
- train_model.py: trains/evaluates models, saves models/digit_model.joblib and models/metrics.json
- digit_recognizer.py: preprocessing, model loading, prediction
- app.py: Streamlit website

## Run
pip install -r requirements.txt
python train_model.py
streamlit run app.py