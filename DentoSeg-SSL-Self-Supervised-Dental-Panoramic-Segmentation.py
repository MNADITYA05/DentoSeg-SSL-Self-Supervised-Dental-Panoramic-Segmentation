import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import layers, models
import pathlib
import natsort
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm, trange  # for progress bars

# Configuration
IMAGE_SHAPE = (224, 224, 1)
BATCH_SIZE = 16
LEARNING_RATE_PRETRAIN = 3e-4
LEARNING_RATE_FINETUNE = 1e-4
WEIGHT_DECAY = 1e-5
TEMPERATURE = 0.1  # For contrastive learning
PROJECTION_DIM = 128
EPOCHS_PRETRAIN = 50
EPOCHS_FINETUNE = 50

# Dataset Class
class DentalDataset:
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path

    def read_paths(self):
        """Read all image and mask paths and sort them naturally"""
        self.images = natsort.natsorted(list(pathlib.Path(self.images_path).glob('*.*')))
        self.masks = natsort.natsorted(list(pathlib.Path(self.masks_path).glob('*.*')))
        return len(self.images), len(self.masks)

    def read_images(self, data_paths, is_mask=False):
        """Load and preprocess images or masks with progress bar"""
        images = []

        # Create progress bar
        pbar = tqdm(data_paths, desc="Loading images" if not is_mask else "Loading masks")

        for img_path in pbar:
            img = cv2.imread(str(img_path), 0)
            img = img / 255.0
            img = cv2.resize(img, (224, 224))
            if is_mask:
                img = np.where(img > 0, 1, 0)
            images.append(img)

        return np.array(images)

    def prepare_data(self, test_ratio=0.2):
        """Prepare the dataset for training"""
        # Read paths
        num_images, num_masks = self.read_paths()
        print(f"Found {num_images} images and {num_masks} masks")

        # Read all images and masks
        all_images = self.read_images(self.images)
        all_masks = self.read_images(self.masks, is_mask=True)

        # Split into train and test
        train_images, test_images, train_masks, test_masks = train_test_split(
            all_images, all_masks, test_size=test_ratio, random_state=42)

        # Add channel dimension if needed
        if len(train_images.shape) == 3:
            train_images = np.expand_dims(train_images, -1)
            test_images = np.expand_dims(test_images, -1)

        if len(train_masks.shape) == 3:
            train_masks = np.expand_dims(train_masks, -1)
            test_masks = np.expand_dims(test_masks, -1)

        print(f"Train: {len(train_images)}, Test: {len(test_images)}")

        return (train_images, train_masks), (test_images, test_masks)

# Data Augmentation for Contrastive Learning
def get_augmentation_layers():
    """Create a model for image augmentation"""
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.GaussianNoise(0.1),
    ])
    return data_augmentation

# Custom contrast adjustment (since RandomContrast might not be available in all TF versions)
def random_contrast(image, lower=0.9, upper=1.1):
    """Apply random contrast adjustment"""
    factor = tf.random.uniform([], lower, upper)
    mean = tf.reduce_mean(image)
    return (image - mean) * factor + mean

# Dice Loss Implementation
def dice_loss(y_true, y_pred, smooth=1):
    """Dice loss for segmentation"""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

# Dice Coefficient Metric
def dice_coefficient(y_true, y_pred, threshold=0.5, smooth=1):
    """
    Dice coefficient metric for segmentation
    Formula: 2*|X∩Y|/(|X|+|Y|)
    """
    # Threshold predictions to get binary masks
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    # Flatten the arrays
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

    # Calculate Dice coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

# Encoder Network
def create_encoder(input_shape=IMAGE_SHAPE):
    """Create the encoder part of the model"""
    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    skip1 = x
    x = layers.MaxPooling2D()(x)  # 112x112

    # Block 2
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    skip2 = x
    x = layers.MaxPooling2D()(x)  # 56x56

    # Block 3
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    skip3 = x
    x = layers.MaxPooling2D()(x)  # 28x28

    # Block 4
    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    skip4 = x
    x = layers.MaxPooling2D()(x)  # 14x14

    # Block 5 (Bottleneck)
    x = layers.Conv2D(1024, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(1024, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Create model
    encoder = tf.keras.Model(inputs=inputs, outputs=[x, skip1, skip2, skip3, skip4], name='encoder')
    return encoder

# Projection Head for Contrastive Learning
def create_projection_head(encoder_output, projection_dim=PROJECTION_DIM):
    """Create projection head for contrastive learning"""
    x = layers.GlobalAveragePooling2D()(encoder_output)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(projection_dim)(x)
    return x

# Contrastive Learning Model
def create_contrastive_model(encoder, projection_dim=PROJECTION_DIM):
    """Create contrastive learning model"""
    # Create two augmentation layers
    augmentation = get_augmentation_layers()

    # Inputs for original images
    inputs = layers.Input(shape=IMAGE_SHAPE)

    # Apply different augmentations to create two views
    aug1 = augmentation(inputs)
    aug2 = augmentation(inputs)

    # Apply custom contrast adjustment
    aug1 = tf.keras.layers.Lambda(lambda x: random_contrast(x))(aug1)
    aug2 = tf.keras.layers.Lambda(lambda x: random_contrast(x))(aug2)

    # Apply encoder to both views
    enc_out1, *_ = encoder(aug1)
    enc_out2, *_ = encoder(aug2)

    # Apply projection head to both outputs
    proj1 = create_projection_head(enc_out1, projection_dim)
    proj2 = create_projection_head(enc_out2, projection_dim)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=[proj1, proj2], name='contrastive_model')
    return model

# Decoder for Segmentation
def create_decoder(encoder_outputs):
    """Create decoder for segmentation"""
    bottleneck, skip1, skip2, skip3, skip4 = encoder_outputs

    # Upsample and merge with skip connections
    # Block 1
    x = layers.Conv2DTranspose(512, 3, strides=2, padding='same')(bottleneck)
    x = layers.Concatenate()([x, skip4])
    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Block 2
    x = layers.Conv2DTranspose(256, 3, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip3])
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Block 3
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip2])
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Block 4
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip1])
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Final output
    x = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)

    return x

# Full Segmentation Model
def create_segmentation_model(encoder):
    """Create segmentation model using pretrained encoder"""
    inputs = layers.Input(shape=IMAGE_SHAPE)

    # Get encoder outputs
    encoder_outputs = encoder(inputs)

    # Create decoder
    segmentation = create_decoder(encoder_outputs)

    # Create model
    segmentation_model = tf.keras.Model(inputs=inputs, outputs=segmentation, name='segmentation_model')

    return segmentation_model

# IoU Metric Implementation
def iou_metric(y_true, y_pred, threshold=0.5):
    """IoU metric for segmentation"""
    # Threshold predictions
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - intersection

    # Calculate IoU
    iou = tf.reduce_mean((intersection + 1e-7) / (union + 1e-7))
    return iou

# NT-Xent Loss Function for Contrastive Learning
def nt_xent_loss(z1, z2, temperature=TEMPERATURE):
    """NT-Xent loss for contrastive learning"""
    # Normalize embeddings along the feature dimension
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)

    # Gather all embeddings
    batch_size = tf.shape(z1)[0]

    # Compute similarity matrix
    z = tf.concat([z1, z2], axis=0)  # 2B x D
    sim = tf.matmul(z, z, transpose_b=True)  # 2B x 2B

    # Create masks for positive pairs
    pos_mask = tf.zeros([2 * batch_size, 2 * batch_size], dtype=tf.float32)
    # Mask for z1 vs z2 (first block vs second block)
    pos_mask = tf.linalg.set_diag(pos_mask, tf.ones([batch_size], dtype=tf.float32), k=batch_size)
    # Mask for z2 vs z1 (second block vs first block)
    pos_mask = tf.linalg.set_diag(pos_mask, tf.ones([batch_size], dtype=tf.float32), k=-batch_size)

    # Compute loss
    sim = sim / temperature

    # We want to maximize similarity between positive pairs, minimize between others
    # For numerical stability, subtract max from each row
    sim_max = tf.reduce_max(sim, axis=1, keepdims=True)
    sim = sim - sim_max

    # Get numerator (positive pairs)
    exp_sim = tf.exp(sim)
    pos_sim = tf.reduce_sum(exp_sim * pos_mask, axis=1)

    # Get denominator (all pairs except self)
    self_mask = tf.eye(2 * batch_size, dtype=tf.float32)
    den_sim = tf.reduce_sum(exp_sim * (1 - self_mask), axis=1)

    # Compute loss
    loss = -tf.reduce_mean(tf.math.log(pos_sim / den_sim + 1e-10))

    return loss


# Fixed ContrastiveTrainer class with progress bar
class ContrastiveTrainer:
    def __init__(self, model, temperature=TEMPERATURE, learning_rate=LEARNING_RATE_PRETRAIN):
        self.model = model
        self.temperature = temperature
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def train_step(self, images):
        with tf.GradientTape() as tape:
            # Forward pass
            z1, z2 = self.model(images, training=True)

            # Compute loss
            loss = nt_xent_loss(z1, z2, self.temperature)

        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def train(self, dataset, epochs):
        # Count total batches
        total_batches = sum(1 for _ in dataset)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            # Initialize progress bar
            progress_bar = tqdm(dataset, total=total_batches, desc=f"Epoch {epoch+1}/{epochs}")

            # Initialize loss for this epoch
            epoch_loss = 0.0
            num_batches = 0

            # Loop through batches
            for batch in progress_bar:
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1

                # Update progress bar
                progress_bar.set_postfix({"loss": f"{loss.numpy():.4f}"})

            # Print metrics
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Average Loss: {avg_loss:.4f}")

# Custom progress bar callback for Keras
class TqdmProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs, verbose=1):
        super(TqdmProgressCallback, self).__init__()
        self.epochs = epochs
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}/{self.epochs}")
        self.progbar = tqdm(total=self.params['steps'],
                           desc=f"Epoch {epoch+1}/{self.epochs}",
                           leave=True)

    def on_batch_end(self, batch, logs=None):
        self.progbar.update(1)
        if logs is not None:
            self.progbar.set_postfix({k: f"{v:.4f}" for k, v in logs.items()})

    def on_epoch_end(self, epoch, logs=None):
        self.progbar.close()
        if self.verbose and logs is not None:
            metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
            print(f"Epoch {epoch+1}/{self.epochs}: {metrics_str}")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Create a proper Grad-CAM heatmap for a segmentation model

    Args:
        img_array: Input image (should be preprocessed)
        model: Trained segmentation model
        last_conv_layer_name: Name of the last conv layer, if None will try to find automatically
        pred_index: Index of output to use for class activation map, if None will use predictions

    Returns:
        Heatmap array
    """
    # If last_conv_layer not specified, find it automatically
    if last_conv_layer_name is None:
        # Find the last convolutional layer
        for layer in reversed(model.layers):
            if 'conv' in layer.name and not 'transpose' in layer.name:
                last_conv_layer_name = layer.name
                print(f"Using {last_conv_layer_name} as the last convolutional layer")
                break

    # First, create a model that maps the input image to the activations
    # of the last conv layer and the output predictions
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # Then, compute the gradient of the output with respect to the output feature map
    with tf.GradientTape() as tape:
        # Cast image to float32
        img_array = tf.cast(img_array, tf.float32)

        # Compute activations of the last conv layer and predictions
        last_conv_layer_output, preds = grad_model(img_array)

        # For segmentation, we want to use the mean of the entire prediction
        # as our target instead of a specific class index
        if pred_index is None:
            if preds.shape[-1] == 1:  # Binary segmentation
                pred_index = 0

        # Take the mean prediction across the entire output (for segmentation)
        pred = preds[:, :, :, pred_index]

        # This is the gradient of the mean prediction with respect to
        # the output feature map of the last conv layer
        grads = tape.gradient(tf.reduce_mean(pred), last_conv_layer_output)

    # Vector of shape (batch_size, features), with one value per channel
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

    # Weight the channels by the gradient importance
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()[0]

    # Apply the weights to the conv layer outputs
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # Average the channels to get the heatmap
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # ReLU to only keep positive contributions
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)

    heatmap = 1.0 - heatmap

    return heatmap

def overlay_gradcam(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on the original image with customizable colormap

    Args:
        img: Original image (grayscale)
        heatmap: Grad-CAM heatmap
        alpha: Transparency factor
        colormap: OpenCV colormap to use (default: COLORMAP_JET)

    Returns:
        Overlaid image
    """
    # Ensure img is normalized to 0-255 range
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Apply colormap to heatmap
    heatmap = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap, colormap)

    # Convert grayscale to RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 1:
        img = cv2.cvtColor(np.squeeze(img), cv2.COLOR_GRAY2RGB)

    # Superimpose the heatmap on original image
    overlaid_img = colored_heatmap * alpha + img * (1 - alpha)
    overlaid_img = np.clip(overlaid_img, 0, 255).astype(np.uint8)

    return overlaid_img

def visualize_results(model, images, masks, num_samples=5):
    """Visualize segmentation results with visualization heatmap"""
    # Predict masks with progress bar
    print("Generating predictions for visualization...")
    pred_progress = tqdm(total=1, desc="Predicting")
    pred_masks = model.predict(images, verbose=0)
    pred_binary = (pred_masks > 0.5).astype(np.float32)
    pred_progress.update(1)
    pred_progress.close()

    # Create figure
    plt.figure(figsize=(20, 5*num_samples))

    # Print progress info for visualization
    print("Generating heatmaps for visualization...")
    heatmap_progress = tqdm(total=num_samples)

    for i in range(num_samples):
        idx = np.random.randint(0, len(images))

        # Prepare the image for heatmap generation
        img_input = images[idx:idx+1]  # Add batch dimension

        try:
            # Generate heatmap with robust error handling
            heatmap = make_gradcam_heatmap(img_input, model)

            # Get original image for visualization
            orig_image = np.squeeze(images[idx])

            # Overlay heatmap on original image
            heatmap_image = overlay_gradcam(orig_image, heatmap)

            # Original image
            plt.subplot(num_samples, 4, i*4+1)
            plt.imshow(orig_image, cmap='gray')
            plt.title('Original Image')
            plt.axis('off')

            # Ground truth mask
            plt.subplot(num_samples, 4, i*4+2)
            plt.imshow(orig_image, cmap='gray')
            plt.imshow(np.squeeze(masks[idx]), alpha=0.5, cmap='Reds')
            plt.title('Ground Truth')
            plt.axis('off')

            # Predicted mask
            plt.subplot(num_samples, 4, i*4+3)
            plt.imshow(orig_image, cmap='gray')
            plt.imshow(np.squeeze(pred_binary[idx]), alpha=0.5, cmap='Greens')
            plt.title('Prediction')
            plt.axis('off')

            # Heatmap visualization
            plt.subplot(num_samples, 4, i*4+4)
            plt.imshow(heatmap_image)
            plt.title('Grad-CAM')
            plt.axis('off')

        except Exception as e:
            print(f"Error generating heatmap for sample {i}: {str(e)}")
            # In case of error, just display without the heatmap
            # Original image
            plt.subplot(num_samples, 3, i*3+1)
            plt.imshow(np.squeeze(images[idx]), cmap='gray')
            plt.title('Original Image')
            plt.axis('off')

            # Ground truth mask
            plt.subplot(num_samples, 3, i*3+2)
            plt.imshow(np.squeeze(images[idx]), cmap='gray')
            plt.imshow(np.squeeze(masks[idx]), alpha=0.5, cmap='Reds')
            plt.title('Ground Truth')
            plt.axis('off')

            # Predicted mask
            plt.subplot(num_samples, 3, i*3+3)
            plt.imshow(np.squeeze(images[idx]), cmap='gray')
            plt.imshow(np.squeeze(pred_binary[idx]), alpha=0.5, cmap='Greens')
            plt.title('Prediction')
            plt.axis('off')

        heatmap_progress.update(1)

    heatmap_progress.close()

    plt.tight_layout()
    plt.savefig('segmentation_results_with_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(20, 5))

    # Plot available metrics with Dice coefficient added
    metrics = ['loss', 'accuracy', 'iou_metric', 'dice_coefficient']
    titles = ['Loss', 'Accuracy', 'IoU', 'Dice Coefficient']

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        if metric in history.history:
            plt.subplot(1, 4, i+1)
            plt.plot(history.history[metric], label='Train')
            val_metric = f'val_{metric}'
            if val_metric in history.history:
                plt.plot(history.history[val_metric], label='Validation')
            plt.title(title)
            plt.xlabel('Epoch')
            plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main Execution
def main():
    # Paths to dataset - update these to match your environment
    images_path = '/kaggle/input/childrens-dental-panoramic-radiographs-dataset/Dental_dataset/Adult tooth segmentation dataset/Panoramic radiography database/images'
    masks_path = '/kaggle/input/childrens-dental-panoramic-radiographs-dataset/Dental_dataset/Adult tooth segmentation dataset/Panoramic radiography database/mask'

    # Create and prepare dataset
    dental_dataset = DentalDataset(images_path, masks_path)
    (train_images, train_masks), (test_images, test_masks) = dental_dataset.prepare_data()

    # Create tf.data.Dataset for contrastive learning
    train_ds = tf.data.Dataset.from_tensor_slices(train_images)
    train_ds = train_ds.shuffle(len(train_images)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Create encoder
    encoder = create_encoder()

    # Create contrastive model
    contrastive_model = create_contrastive_model(encoder)

    # Print model summary
    print("Contrastive Model Summary:")
    contrastive_model.summary()

    # Train contrastive model
    print("\nPretraining contrastive model...")
    trainer = ContrastiveTrainer(contrastive_model)
    trainer.train(train_ds, epochs=EPOCHS_PRETRAIN)

    # Create segmentation model
    print("\nCreating segmentation model...")
    segmentation_model = create_segmentation_model(encoder)

    # Print model summary
    print("Segmentation Model Summary:")
    segmentation_model.summary()

    # Compile segmentation model with dice coefficient added as a metric
    segmentation_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINETUNE),
        loss=dice_loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            iou_metric,
            dice_coefficient  # Added Dice coefficient metric
        ]
    )

    # Create validation split
    val_split = int(0.2 * len(train_images))
    val_images = train_images[:val_split]
    val_masks = train_masks[:val_split]
    train_images_split = train_images[val_split:]
    train_masks_split = train_masks[val_split:]

    # Train segmentation model with custom progress bar
    print("\nTraining segmentation model...")
    history = segmentation_model.fit(
        train_images_split, train_masks_split,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_FINETUNE,
        validation_data=(val_images, val_masks),
        callbacks=[
            TqdmProgressCallback(epochs=EPOCHS_FINETUNE),
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.keras', save_best_only=True, monitor='val_dice_coefficient', mode='max', verbose=1
            )
        ],
        verbose=0  # Turn off default progress bar
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_progress = tqdm(total=1, desc="Evaluating")
    results = segmentation_model.evaluate(test_images, test_masks, verbose=0)
    test_progress.update(1)
    test_progress.close()

    # Print evaluation results
    metric_names = ['Loss', 'Accuracy', 'Precision', 'Recall', 'IoU', 'Dice Coefficient']
    for name, value in zip(metric_names, results):
        print(f"Test {name}: {value:.4f}")

    # Save model
    print("\nSaving model...")
    save_progress = tqdm(total=1, desc="Saving model")
    segmentation_model.save('dental_segmentation_model.h5')
    save_progress.update(1)
    save_progress.close()

    # Visualize results
    print("\nVisualizing results...")
    visualize_results(segmentation_model, test_images, test_masks)

    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history)

    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    # Try to catch any errors and provide helpful messages
    try:
        main()
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have the tqdm package installed (pip install tqdm)")
        print("2. Check that your dataset paths are correct")
        print("3. Ensure you have enough memory for the model and dataset")
        print("4. Try reducing batch size or model size if you're running out of memory")
        raise
