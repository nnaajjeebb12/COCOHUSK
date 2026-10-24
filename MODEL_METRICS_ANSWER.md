# Model metrics for the client (F1, precision, recall)

Client request: *"need na po pala namin i lagay yung mga f1 score, precision yung mga ganon po"*

## TL;DR

- We now have real F1, precision, recall, accuracy and a confusion matrix.
  The files to send are in [evaluation/](evaluation/).
- **Test accuracy: 92%, macro F1: 0.917** on 50 photos the model never saw
  during training.
- Over 6 different random train/test splits, accuracy ranged **92% to 100%
  (average about 95%)**, so the result is stable.
- The model is a **CNN** (3 convolutional layers). It is **not**
  EfficientNetB0, which the README wrongly said before (now fixed).

## Main results (test set, 50 unseen images)

| Class      | Precision | Recall | F1-score | Test images |
|------------|-----------|--------|----------|-------------|
| Immature   | 0.8235    | 0.9333 | 0.8750   | 15          |
| Mature     | 0.9333    | 0.8235 | 0.8750   | 17          |
| Overmature | 1.0000    | 1.0000 | 1.0000   | 18          |
| **Macro avg**    | **0.9190** | **0.9190** | **0.9167** | 50 |
| **Weighted avg** | **0.9244** | **0.9200** | **0.9200** | 50 |

**Accuracy: 0.9200 (92%)**

Confusion matrix (rows = actual, columns = predicted):

|                 | Pred. immature | Pred. mature | Pred. overmature |
|-----------------|----------------|--------------|------------------|
| **Immature**    | 14             | 1            | 0                |
| **Mature**      | 3              | 14           | 0                |
| **Overmature**  | 0              | 0            | 18               |

All 4 mistakes are between *immature* and *mature*. Overmature is never
confused with anything. That makes sense, since immature and mature husks look
the most alike.

### Stability across splits (optional, adds credibility)

Same model, retrained on 5 more random splits:

| Metric    | Mean (6 runs) | Range       |
|-----------|---------------|-------------|
| Accuracy  | 0.953         | 0.92 – 1.00 |
| Macro F1  | 0.953         | 0.917 – 1.00 |

## Files to send the student

| File | What it is |
|------|------------|
| [evaluation/test_report.txt](evaluation/test_report.txt) | Full metric table plus dataset counts |
| [evaluation/confusion_matrix_test.png](evaluation/confusion_matrix_test.png) | Confusion matrix figure (good for the paper) |
| [evaluation/training_curves.png](evaluation/training_curves.png) | Training vs. validation accuracy/loss per epoch |
| [evaluation/metrics.json](evaluation/metrics.json) | Same numbers in machine-readable form |

## Ready-to-send message (Taglish)

> Hi po! Eto na po yung evaluation results ng model (CNN):
>
> **Accuracy: 92%**
>
> | Class | Precision | Recall | F1-score |
> |---|---|---|---|
> | Immature | 0.82 | 0.93 | 0.88 |
> | Mature | 0.93 | 0.82 | 0.88 |
> | Overmature | 1.00 | 1.00 | 1.00 |
> | **Average (macro)** | **0.92** | **0.92** | **0.92** |
>
> Tinest po ito sa 50 images na hindi ginamit sa training (70% train,
> 15% validation, 15% test, total 333 unique images). Attached din po yung
> confusion matrix at training graph. Karamihan ng mali ay sa pagitan ng
> immature at mature kasi magkahawig sila; yung overmature ay 100% tama.

## What the metrics mean (in case the student asks)

- **Accuracy**: out of all test photos, how many were classified correctly.
- **Precision**: when the model says "mature", how often it's really mature.
  High precision means few false alarms.
- **Recall**: of all the truly mature husks, how many the model caught.
  High recall means few misses.
- **F1-score**: the balance of precision and recall in one number (their
  harmonic mean). Use it when you want a single score per class.
- **Macro avg**: the plain average across the 3 classes. **Weighted avg**: the
  average weighted by how many test images each class has.
- **Confusion matrix**: shows exactly which class gets mistaken for which.

## Important notes (for us, not necessarily for the client)

1. **Why not 100%?** If you test the app's current model on the
   `images/*-untrained` folders, you get 100% on everything. That number is
   **not valid**. Those folders are mostly byte-for-byte copies of the
   training photos, and in fact every photo in `images/` is stored about 4
   times. Testing on photos the model already saw is cheating, and a thesis
   panel that notices would reject it. After removing duplicates there are
   **333 unique photos** (101 immature, 112 mature, 120 overmature).
2. **Which model these numbers come from.** The script that trained the app's
   bundled `.tflite` isn't in the repo, and we can't tell which photos it
   trained on, so it can't be fairly tested. The numbers above come from
   **the same CNN architecture**, retrained by
   [train_and_evaluate.py](train_and_evaluate.py) on a clean split. That's
   the standard way to report metrics. If you want the app to run exactly
   the model these numbers describe, convert `evaluation/model_clean_split.keras`
   with `convert_to_tflite.py` and replace `assets/model/coconut_husk_quality_model.tflite`.
   **This hasn't been done yet. The app model is unchanged.**
3. **Small test set.** 50 images is small, so one mistake moves accuracy by
   2%. That's why the 6-run range (92–100%) is worth mentioning. More photos
   would make the numbers more reliable.
4. **These metrics use the model's top prediction only.** The app also has a
   "reject / not sure" layer (confidence and margin thresholds) on top. These
   numbers don't include it.

## How to reproduce

```
pip install tensorflow numpy pillow scikit-learn matplotlib
python train_and_evaluate.py
```

Takes about a minute on a normal CPU. The results go to `evaluation/`.
Random seed is 42. Changing `SEED` in the script gives the other splits.
