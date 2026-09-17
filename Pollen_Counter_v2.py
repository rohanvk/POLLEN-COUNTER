"""
Pollen Counter v2 — Improved CNN Regression Model
===================================================
Counts pollen grains in microscope images using transfer learning
with EfficientNetB0. Designed to run in Google Colab with Google Drive.

Improvements over v1:
  - Transfer learning (EfficientNetB0 pretrained on ImageNet)
  - Proper steps_per_epoch and validation_steps
  - Dropout + GlobalAveragePooling for regularization
  - EarlyStopping, ReduceLROnPlateau, ModelCheckpoint callbacks
  - Lower learning rate for stable training
  - MAE metric for interpretable error in pollen count units
  - Fine-tuning phase after initial frozen training
  - Comprehensive evaluation: scatter plot, sample predictions, error histogram
  - TensorFlow.js export for web deployment
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

# ---------------------------------------------------------------------------
# Configuration — adjust these as needed
# ---------------------------------------------------------------------------
DATA_DIR = "/content/drive/MyDrive/Pollen Training Images"
CSV_FILE = "/content/image_counts.csv"
IMG_SIZE = (224, 224)  # EfficientNet default input size
BATCH_SIZE = 16
INITIAL_EPOCHS = 50   # frozen backbone
FINETUNE_EPOCHS = 30  # unfrozen top layers
INITIAL_LR = 1e-3
FINETUNE_LR = 1e-5
MODEL_SAVE_PATH = "/content/pollen_counter_best.keras"
TFJS_OUTPUT_DIR = "tfjs_pollen_model"


# ===================================================================
# 1. Mount Google Drive (Colab only — skip if running locally)
# ===================================================================
def mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        print("Google Drive mounted.")
    except ImportError:
        print("Not running in Colab — skipping drive mount.")


# ===================================================================
# 2. Load and split data
# ===================================================================
def load_data(csv_file, data_dir, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_file).dropna()
    df["filename"] = df["filename"].apply(lambda x: os.path.join(data_dir, x))

    # Ensure the count column is float for regression
    df["count"] = df["count"].astype(float)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    print(f"Total images:      {len(df)}")
    print(f"Training images:   {len(train_df)}")
    print(f"Test images:       {len(test_df)}")
    print(f"Count range:       {df['count'].min():.0f} – {df['count'].max():.0f}")

    return train_df, test_df


# ===================================================================
# 3. Create data generators
# ===================================================================
def create_generators(train_df, test_df, img_size, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
        validation_split=0.2,  # carve out 20% of training for validation
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filename",
        y_col="count",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="raw",
        seed=42,
        subset="training",
    )

    val_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filename",
        y_col="count",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="raw",
        seed=42,
        subset="validation",
    )

    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="filename",
        y_col="count",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="raw",
        shuffle=False,
        seed=42,
    )

    print(f"\nTrain samples:      {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples:       {test_gen.samples}")

    return train_gen, val_gen, test_gen


# ===================================================================
# 4. Build model with transfer learning
# ===================================================================
def build_model(img_size, freeze_backbone=True):
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*img_size, 3),
    )
    base_model.trainable = not freeze_backbone

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dropout(0.4),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1),  # linear output for regression
    ])

    return model, base_model


# ===================================================================
# 5. Training — Phase 1: frozen backbone
# ===================================================================
def train_frozen(model, train_gen, val_gen, epochs, lr, save_path):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )

    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_mae",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            save_path,
            monitor="val_mae",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # Let Keras compute steps from generator length automatically
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    return history


# ===================================================================
# 6. Training — Phase 2: fine-tune top layers of backbone
# ===================================================================
def finetune(model, base_model, train_gen, val_gen, epochs, lr, save_path,
             previous_history):
    # Unfreeze the top 30% of layers in the backbone
    num_layers = len(base_model.layers)
    freeze_until = int(num_layers * 0.7)
    base_model.trainable = True
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    trainable = sum(1 for l in model.layers for w in l.trainable_weights)
    total = sum(1 for l in model.layers for w in l.weights)
    print(f"\nFine-tuning: {trainable}/{total} weight arrays are trainable")
    print(f"Unfreezing layers {freeze_until}–{num_layers} of backbone\n")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_mae",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.5,
            patience=4,
            min_lr=1e-8,
            verbose=1,
        ),
        ModelCheckpoint(
            save_path,
            monitor="val_mae",
            save_best_only=True,
            verbose=1,
        ),
    ]

    initial_epoch = len(previous_history.history["loss"])

    history = model.fit(
        train_gen,
        epochs=initial_epoch + epochs,
        initial_epoch=initial_epoch,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    return history


# ===================================================================
# 7. Evaluation
# ===================================================================
def evaluate(model, test_gen, test_df):
    print("\n" + "=" * 50)
    print("EVALUATION")
    print("=" * 50)

    # Get predictions
    preds = model.predict(test_gen).flatten()
    actuals = test_df["count"].values[: len(preds)]

    # Metrics
    mse = np.mean((actuals - preds) ** 2)
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)
    rmse = np.sqrt(mse)

    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}  (average error in pollen count)")
    print(f"R²:   {r2:.4f}")
    print()

    # --- Plot 1: Predicted vs Actual scatter ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.scatter(actuals, preds, alpha=0.6, edgecolors="k", linewidths=0.5)
    max_val = max(actuals.max(), preds.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "r--", label="Perfect prediction")
    ax.set_xlabel("Actual Pollen Count")
    ax.set_ylabel("Predicted Pollen Count")
    ax.set_title(f"Predicted vs Actual  (R² = {r2:.3f})")
    ax.legend()
    ax.set_aspect("equal")

    # --- Plot 2: Error histogram ---
    ax = axes[1]
    errors = preds - actuals
    ax.hist(errors, bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="r", linestyle="--")
    ax.set_xlabel("Prediction Error (pred − actual)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Error Distribution  (MAE = {mae:.2f})")

    # --- Plot 3: Sorted comparison ---
    ax = axes[2]
    sort_idx = np.argsort(actuals)
    ax.plot(actuals[sort_idx], label="Actual", marker="o", markersize=3)
    ax.plot(preds[sort_idx], label="Predicted", marker="x", markersize=3)
    ax.set_xlabel("Test Sample (sorted by actual count)")
    ax.set_ylabel("Pollen Count")
    ax.set_title("Actual vs Predicted (sorted)")
    ax.legend()

    plt.tight_layout()
    plt.savefig("/content/pollen_evaluation.png", dpi=150)
    plt.show()
    print("Saved evaluation plots to /content/pollen_evaluation.png")

    return preds, actuals


# ===================================================================
# 8. Plot training history
# ===================================================================
def plot_history(*histories):
    loss, val_loss, mae_vals, val_mae = [], [], [], []
    for h in histories:
        loss.extend(h.history["loss"])
        val_loss.extend(h.history.get("val_loss", []))
        mae_vals.extend(h.history["mae"])
        val_mae.extend(h.history.get("val_mae", []))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(loss, label="Train Loss (MSE)")
    if val_loss:
        ax1.plot(val_loss, label="Val Loss (MSE)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(mae_vals, label="Train MAE")
    if val_mae:
        ax2.plot(val_mae, label="Val MAE")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE (pollen count)")
    ax2.set_title("Mean Absolute Error")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("/content/pollen_training_history.png", dpi=150)
    plt.show()
    print("Saved training history to /content/pollen_training_history.png")


# ===================================================================
# 9. Export to TensorFlow.js
# ===================================================================
def export_tfjs(model_path, output_dir):
    print("\nExporting to TensorFlow.js...")
    os.system("pip install tensorflowjs -q")
    os.system(
        f"tensorflowjs_converter --input_format=keras "
        f"{model_path} {output_dir}"
    )

    # Zip for download
    zip_path = f"{output_dir}.zip"
    os.system(f"zip -r {zip_path} {output_dir}")
    print(f"TF.js model saved to {zip_path}")

    try:
        from google.colab import files
        files.download(zip_path)
    except ImportError:
        print(f"Download the zip manually: {zip_path}")


# ===================================================================
# Main
# ===================================================================
def main():
    mount_drive()

    # Load data
    train_df, test_df = load_data(CSV_FILE, DATA_DIR)

    # Create generators
    train_gen, val_gen, test_gen = create_generators(
        train_df, test_df, IMG_SIZE, BATCH_SIZE
    )

    # Build model
    model, base_model = build_model(IMG_SIZE, freeze_backbone=True)

    # Phase 1: Train with frozen backbone
    print("\n" + "=" * 50)
    print("PHASE 1: Training with frozen backbone")
    print("=" * 50)
    history1 = train_frozen(
        model, train_gen, val_gen,
        epochs=INITIAL_EPOCHS,
        lr=INITIAL_LR,
        save_path=MODEL_SAVE_PATH,
    )

    # Phase 2: Fine-tune top layers
    print("\n" + "=" * 50)
    print("PHASE 2: Fine-tuning top backbone layers")
    print("=" * 50)
    history2 = finetune(
        model, base_model, train_gen, val_gen,
        epochs=FINETUNE_EPOCHS,
        lr=FINETUNE_LR,
        save_path=MODEL_SAVE_PATH,
        previous_history=history1,
    )

    # Plot training history
    plot_history(history1, history2)

    # Evaluate on held-out test set
    preds, actuals = evaluate(model, test_gen, test_df)

    # Export to TF.js
    export_tfjs(MODEL_SAVE_PATH, TFJS_OUTPUT_DIR)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
