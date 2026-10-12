# Handwritten Digit Recogniser

Draw a digit (0-9) on a web canvas and a PCA + SVM model trained on MNIST recognises it.

## Files
- `train_model.py`: trains/evaluates models, saves `models/digit_model.joblib` and `models/metrics.json`
- `digit_recognizer.py`: preprocessing, model loading, prediction
- `app.py`: Streamlit website

## Run
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
<button block onClick={() => GenUI.copy("# Handwritten Digit Recogniser\n\nDraw a digit (0-9) on a web canvas and a PCA + SVM model trained on MNIST recognises it.\n\n## Files\n- `train_model.py`: trains/evaluates models, saves `models/digit_model.joblib` and `models/metrics.json`\n- `digit_recognizer.py`: preprocessing, model loading, prediction\n- `app.py`: Streamlit website\n\n## Run\n```bash\npip install -r requirements.txt\npython train_model.py\nstreamlit run app.py\n```")}><icon name="copy" size="xs" /> Copy README content</button>
</box>

Then press **Ctrl + S** to save it.

## Step 14 — Check the conflict

Return to the terminal and run:

```bash
git status