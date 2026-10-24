# train_and_evaluate.py
# Trains the same simple CNN as the bundled model on a clean, de-duplicated
# train/val/test split of images/, then reports accuracy, precision, recall,
# F1 and a confusion matrix on the held-out test split.
#
# Why: the images/*-untrained folders are mostly copies of the training
# images, so evaluating on them gives a meaningless 100%. Here duplicates and
# near-duplicates are grouped first, and each group goes to exactly one split.
#
# Usage:  python train_and_evaluate.py
# Output: evaluation/ (report, confusion matrix, training curves, model)
# Requires: tensorflow, numpy, pillow, scikit-learn, matplotlib

import json
import pathlib
import random

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix

keras = tf.keras
layers = tf.keras.layers

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
SPLIT = (0.70, 0.15, 0.15)  # train / val / test
NEAR_DUP_BITS = 5  # dHash distance (out of 256) treated as the same photo
OUT_DIR = pathlib.Path("evaluation")

CLASS_FOLDERS = {
    "immature": ["images/immature", "images/immature-untrained"],
    "mature": ["images/mature", "images/mature-untrained"],
    "overmature": ["images/overmature", "images/over-untrained"],
}


def dhash(img):
    a = np.asarray(img.convert("L").resize((17, 16)), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def load_class(folders):
    """Loads every image of one class, merged into near-duplicate groups."""
    imgs, hashes = [], []
    for folder in folders:
        for f in sorted(pathlib.Path(folder).glob("*.jpg")):
            img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            hashes.append(dhash(img))
            imgs.append(np.asarray(img.resize(IMG_SIZE, Image.BILINEAR), dtype=np.uint8))

    # Union-find over images whose hashes are within NEAR_DUP_BITS.
    h = np.array(hashes)
    parent = list(range(len(h)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(h)):
        close = np.nonzero((h[i + 1:] != h[i]).sum(1) <= NEAR_DUP_BITS)[0] + i + 1
        for j in close:
            parent[find(j)] = find(i)

    groups = {}
    for i in range(len(h)):
        groups.setdefault(find(i), []).append(i)
    # One representative per group: duplicates add no information.
    return [imgs[g[0]] for g in groups.values()], len(imgs)


def split_data():
    rng = random.Random(SEED)
    splits = {"train": ([], []), "val": ([], []), "test": ([], [])}
    counts = {}
    for label_idx, (label, folders) in enumerate(CLASS_FOLDERS.items()):
        unique, total = load_class(folders)
        rng.shuffle(unique)
        n_train = round(len(unique) * SPLIT[0])
        n_val = round(len(unique) * SPLIT[1])
        parts = {
            "train": unique[:n_train],
            "val": unique[n_train:n_train + n_val],
            "test": unique[n_train + n_val:],
        }
        counts[label] = {"files": total, "unique": len(unique),
                         **{k: len(v) for k, v in parts.items()}}
        for k, v in parts.items():
            splits[k][0].extend(v)
            splits[k][1].extend([label_idx] * len(v))
    arrays = {k: (np.array(x, dtype=np.float32), np.array(y)) for k, (x, y) in splits.items()}
    return arrays, counts


def create_model(num_classes):
    # Same architecture as the bundled app model (see train_model.py).
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="coconut_husk_maturity_model")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    tf.keras.utils.set_random_seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    labels = list(CLASS_FOLDERS)

    data, counts = split_data()
    print("Dataset after de-duplication:")
    print(json.dumps(counts, indent=2))

    model = create_model(len(labels))
    history = model.fit(
        *data["train"],
        validation_data=data["val"],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True)],
    )

    x_test, y_test = data["test"]
    y_pred = model.predict(x_test, verbose=0).argmax(1)
    report = classification_report(y_test, y_pred, target_names=labels, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nTest set ({len(y_test)} images, never seen in training):\n")
    print(report)
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(cm)

    (OUT_DIR / "test_report.txt").write_text(
        f"Dataset counts:\n{json.dumps(counts, indent=2)}\n\n"
        f"Test set ({len(y_test)} images):\n\n{report}\n"
        f"Confusion matrix (rows = actual, columns = predicted; order {labels}):\n{cm}\n")
    (OUT_DIR / "metrics.json").write_text(json.dumps({
        "counts": counts,
        "report": classification_report(y_test, y_pred, target_names=labels, output_dict=True),
        "confusion_matrix": cm.tolist(),
        "epochs_trained": len(history.history["loss"]),
    }, indent=2))
    model.save(OUT_DIR / "model_clean_split.keras")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues", colorbar=False)
    plt.title(f"Confusion matrix (test set, n={len(y_test)})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix_test.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, ["accuracy", "loss"]):
        ax.plot(history.history[key], label="train")
        ax.plot(history.history[f"val_{key}"], label="validation")
        ax.set_title(key.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_curves.png", dpi=150)
    print(f"\nSaved results to {OUT_DIR}/")


if __name__ == "__main__":
    main()
