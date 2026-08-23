# train_model.py (simplified)
import tensorflow as tf
import pathlib
keras = tf.keras
layers = tf.keras.layers


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = pathlib.Path("dataset")

def create_datasets():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR / "train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR / "val",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    class_names = train_ds.class_names
    print("Class names:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names

def create_model(num_classes):
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=IMG_SIZE + (3,),
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model

def main():
    train_ds, val_ds, class_names = create_datasets()
    num_classes = len(class_names)
    model, base_model = create_model(num_classes)

    model.fit(train_ds, validation_data=val_ds, epochs=10)

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=10)

    model.save("coconut_husk_quality_model.h5")
    with open("class_names.txt", "w") as f:
        for name in class_names:
            f.write(name + "\n")

if __name__ == "__main__":
    main()