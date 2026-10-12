"""
Handwritten Digit Recogniser with a Drawing Canvas
Machine Learning (25SC2107E) - Sec 1, Team 19

Algorithm : PCA (reduce features) + SVM with RBF kernel   (Al-Hamadani, 2023)
Dataset   : MNIST, 70,000 handwritten digit images (60,000 train / 10,000 test)
Input     : a digit (0-9) drawn with the mouse in MS Paint or Microsoft Whiteboard
            and saved as an image (PNG / JPG)

Install   : pip install scikit-learn numpy pillow joblib
Step 1    : python digit_recognizer.py train            (needs internet the first time)
            python digit_recognizer.py train --fast     (quick demo, uses 10,000 images)
Step 2    : python digit_recognizer.py predict my_digit.png

The program shows an error message if the drawing is blank, too small, has more than
one digit, or is not a valid image. It warns when it is not sure.

Tips for drawing: use a plain white canvas, a black pen of medium thickness,
and draw ONE big digit that fills most of the canvas.
"""
import os
import sys
import numpy as np
import joblib
from PIL import Image, ImageFilter
from sklearn.datasets import fetch_openml, load_digits
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_FILE = "digit_model.joblib"
PREVIEW_FILE = "what_the_model_sees.png"
MIN_CONFIDENCE = 0.60          # below this, the program warns that it may not be a clear digit


def softmax(a):
    a = a - a.max(axis=-1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------
# Module 1 - Data: load MNIST (falls back to a small built-in set)
# ---------------------------------------------------------------
def load_data():
    print("Step 1/4: loading MNIST. The first time it downloads about 15 MB,")
    print("          so it can take a few minutes with no new lines. Please wait...")
    try:
        X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
        X = X.astype("float32") / 255.0              # scale pixels to 0-1
        y = y.astype(int)
        return X[:60000], X[60000:], y[:60000], y[60000:], 28
    except Exception as error:
        print("Could not download MNIST (%s)." % type(error).__name__)
        print("Using the small built-in 8x8 digits set instead (for testing only).")
        data = load_digits()
        X = data.data / 16.0
        Xtr, Xte, ytr, yte = train_test_split(X, data.target, test_size=0.2, random_state=0)
        return Xtr, Xte, ytr, yte, 8


# ---------------------------------------------------------------
# Module 2 + 3 - Processing and Core logic: PCA + SVM
# ---------------------------------------------------------------
def train(fast=False):
    Xtr, Xte, ytr, yte, size = load_data()
    if fast and len(Xtr) > 10000:
        Xtr, ytr = Xtr[:10000], ytr[:10000]
    print("Training images: %d | Test images: %d | Image size: %dx%d"
          % (len(Xtr), len(Xte), size, size))

    parts = 50 if size == 28 else 20
    pca = PCA(n_components=parts, random_state=0).fit(Xtr)       # reduce the features
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    print("Step 2/4: PCA done: %d pixels -> %d features" % (Xtr.shape[1], parts))

    print("Step 3/4: training the SVM (RBF kernel)... this can take a few minutes. Please wait...")
    svm = SVC(kernel="rbf", C=5, gamma="scale").fit(Ztr, ytr)

    baseline = LogisticRegression(max_iter=1000).fit(Ztr, ytr)   # simple baseline

    print("Step 4/4: testing the model...")
    # Evaluation: accuracy, precision, recall, confusion matrix
    pred = svm.predict(Zte)
    print("\nSVM accuracy      : %.2f%%" % (100 * accuracy_score(yte, pred)))
    print("Baseline (Logistic Regression) accuracy: %.2f%%"
          % (100 * accuracy_score(yte, baseline.predict(Zte))))
    print("\nPrecision / recall per digit:\n", classification_report(yte, pred, digits=3))
    print("Confusion matrix (rows = real digit, columns = predicted):\n",
          confusion_matrix(yte, pred))

    # Make the confidence score meaningful: pick the scale that best matches the real results
    scores = svm.decision_function(Zte)
    half = len(yte) // 2
    best_scale, best_loss = 1.0, 1e9
    for scale in [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 14]:
        chance = softmax(scores[:half] * scale)[np.arange(half), yte[:half]]
        loss = -np.log(chance + 1e-9).mean()
        if loss < best_loss:
            best_scale, best_loss = scale, loss
    print("Confidence scale chosen:", best_scale)

    joblib.dump({"pca": pca, "svm": svm, "size": size, "scale": best_scale}, MODEL_FILE)
    print("\nDONE. Model saved as", MODEL_FILE, "- now run: python digit_recognizer.py predict your_image.png")


# ---------------------------------------------------------------
# Pre-processing: turn a Paint / Whiteboard drawing into what the model expects
# ---------------------------------------------------------------
def adjust_stroke(ink, target):
    """Make the pen stroke about as thick as in MNIST digits."""
    mask = ink > 100
    inner = mask.copy()
    inner[1:] &= mask[:-1]
    inner[:-1] &= mask[1:]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    edge = (mask & ~inner).sum()
    width = 2.0 * mask.sum() / max(int(edge), 1)             # approx. stroke width
    img = Image.fromarray(ink.astype("uint8"))
    if width < target:                                       # too thin -> thicken
        r = int(round((target - width) / 2))
        if r >= 1:
            img = img.filter(ImageFilter.MaxFilter(2 * r + 1))
    elif width > 2.0 * target:                               # too thick -> thin
        r = int(round((width - 1.5 * target) / 2))
        if r >= 1:
            img = img.filter(ImageFilter.MinFilter(2 * r + 1))
    return np.array(img, dtype=float)


def check_single_digit(ink, full):
    """Reject drawings that are not ONE digit-sized drawing."""
    h, w = ink.shape
    if max(h, w) < max(10, 0.08 * full):
        raise ValueError("The drawing is too small. Draw ONE bigger digit.")
    if w > 1.5 * h:
        raise ValueError("The drawing is too wide. Draw only ONE digit "
                         "(not several digits, letters or symbols).")
    used = np.where((ink > 100).any(axis=0))[0]                 # columns that contain ink
    if len(used) > 1 and np.diff(used).max() - 1 > 0.18 * h:    # big empty gap in the middle
        raise ValueError("More than one digit (or symbol) was found. Draw only ONE digit, 0 to 9.")


def prepare_image(path, size):
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):                      # transparent background -> white
        img = img.convert("RGBA")
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, img)
    gray = np.array(img.convert("L"), dtype=float)

    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    ink = 255.0 - gray if np.median(border) > 127 else gray   # digit must be bright, background dark
    ink = (ink - ink.min()) / max(ink.max() - ink.min(), 1) * 255

    rows, cols = np.where(ink > 100)                          # crop to the digit
    if len(rows) == 0:
        raise ValueError("No drawing found. Draw a dark digit on a white background.")
    full = max(ink.shape)
    ink = ink[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    check_single_digit(ink, full)

    h, w = ink.shape                                          # keep it small, then fix the stroke
    scale = 100.0 / max(h, w)
    ink = np.array(Image.fromarray(ink.astype("uint8")).resize(
        (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS), dtype=float)
    ink = adjust_stroke(ink, target=0.11 * max(ink.shape))

    box = round(size * (20 / 28 if size == 28 else 0.85))     # MNIST digits fill 20 of 28 pixels
    h, w = ink.shape
    scale = box / max(h, w)
    small = np.array(Image.fromarray(ink.astype("uint8")).resize(
        (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS), dtype=float)

    ys, xs = np.indices(small.shape)                          # centre the digit by its centre of mass
    cy, cx = (ys * small).sum() / small.sum(), (xs * small).sum() / small.sum()
    top = int(np.clip(round(size / 2 - cy), 0, size - small.shape[0]))
    left = int(np.clip(round(size / 2 - cx), 0, size - small.shape[1]))
    canvas = np.zeros((size, size))
    canvas[top:top + small.shape[0], left:left + small.shape[1]] = small
    return canvas / 255.0


# ---------------------------------------------------------------
# Module 4 - Interface: read the saved drawing, predict, show result
# ---------------------------------------------------------------
def predict(path):
    model = joblib.load(MODEL_FILE)
    picture = prepare_image(path, model["size"])
    Image.fromarray((picture * 255).astype("uint8")).resize((140, 140)).save(PREVIEW_FILE)

    features = model["pca"].transform(picture.reshape(1, -1))
    digit = int(model["svm"].predict(features)[0])
    scores = model["svm"].decision_function(features)[0]
    share = softmax(scores * model.get("scale", 1.0))          # confidence score

    print("\nPredicted digit : %d   (confidence about %.0f%%)" % (digit, 100 * share[digit]))
    print("Other guesses   :", ", ".join(
        "%d (%.0f%%)" % (d, 100 * share[d]) for d in np.argsort(share)[::-1][1:3]))
    if share[digit] < MIN_CONFIDENCE:
        print("WARNING: low confidence. This may not be a clear single digit (0-9). "
              "Try drawing it again.")
    print("Saved '%s' - this is the image the model saw." % PREVIEW_FILE)


def run_predict(path):
    """Predict one image and show a clear message if something is wrong."""
    path = path.strip().strip('"')
    if not os.path.isfile(MODEL_FILE):
        print("Error: the trained model was not found. Run this first: python digit_recognizer.py train")
    elif not os.path.isfile(path):
        print("Error: the image '%s' was not found. Check the name and the folder." % path)
    else:
        try:
            predict(path)
        except ValueError as error:
            print("Error:", error)
        except OSError:
            print("Error: '%s' is not a valid image. Save your drawing as PNG or JPG." % path)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["train"]:
        train(fast="--fast" in args)
    elif args[:1] == ["predict"]:
        if len(args) > 1:
            run_predict(args[1])
        else:
            while True:                                       # ask for images until Enter is pressed
                name = input("\nPath of your drawing (Enter to quit): ")
                if not name.strip():
                    break
                run_predict(name)
    else:
        print(__doc__)