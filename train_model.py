# train_model.py (key parts only)

import pathlib
import tensorflow as tf

keras = tf.keras
layers = tf.keras.layers

IMG_SIZE = (224, 224)
BATCH_SIZE = 8  # smaller batch for small dataset
DATA_DIR = pathlib.Path("dataset")


def create_datasets():
    train_ds = keras.preprocessing.image_dataset_from_directory(
        DATA_DIR / "train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    # For now (for debugging), use train as val too; later create real val set
    val_ds = keras.preprocessing.image_dataset_from_directory(
        DATA_DIR / "train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    class_names = train_ds.class_names
    print("Class names:", class_names)

    autotune = tf.data.AUTOTUNE
    train_ds = (
        train_ds.cache()
        .shuffle(100)
        .prefetch(buffer_size=autotune)
    )
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


def create_model(num_classes: int):
    # Very simple CNN, no pretrained weights, no downloads
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

    model = keras.Model(inputs, outputs, name="coconut_husk_quality_model")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def main():
    print("TensorFlow version:", tf.__version__)

    train_ds, val_ds, class_names = create_datasets()
    num_classes = len(class_names)

    model = create_model(num_classes)

    print("\n=== Training standalone CNN (no external weights) ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,  # adjust as you get more data
    )

    model_path = "coconut_husk_quality_model.h5"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    class_names_path = "class_names.txt"
    with open(class_names_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"Class names saved to {class_names_path}")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()