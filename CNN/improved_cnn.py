import tensorflow as tf
from tensorflow.keras import datasets, layers, models, mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

# Ativa mixed precision
mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

# Verifica GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU detectada:", gpus[0])
    # Configura para usar apenas a memória necessária
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️ GPU não detectada. Usando CPU.")
    # Desativa mixed precision para CPU
    mixed_precision.set_global_policy('float32')

# Carrega dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Função CutMix

def get_cutmix_box(img_shape, lam):
    img_h = tf.cast(img_shape[0], tf.float32)
    img_w = tf.cast(img_shape[1], tf.float32)
    cut_rat = tf.math.sqrt(1. - lam)
    cut_w = tf.cast(img_w * cut_rat, tf.int32)
    cut_h = tf.cast(img_h * cut_rat, tf.int32)

    cx = tf.random.uniform(shape=(), minval=0, maxval=tf.cast(img_w, tf.int32), dtype=tf.int32)
    cy = tf.random.uniform(shape=(), minval=0, maxval=tf.cast(img_h, tf.int32), dtype=tf.int32)

    bbx1 = tf.clip_by_value(cx - cut_w // 2, 0, tf.cast(img_w, tf.int32))
    bby1 = tf.clip_by_value(cy - cut_h // 2, 0, tf.cast(img_h, tf.int32))
    bbx2 = tf.clip_by_value(cx + cut_w // 2, 0, tf.cast(img_w, tf.int32))
    bby2 = tf.clip_by_value(cy + cut_h // 2, 0, tf.cast(img_h, tf.int32))

    return bbx1, bby1, bbx2, bby2

def cutmix(images, labels, alpha=1.0):
    batch_size = tf.shape(images)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    lam = tfp.distributions.Beta(alpha, alpha).sample()
    bbx1, bby1, bbx2, bby2 = get_cutmix_box(tf.shape(images)[1:3], lam)

    img_area = tf.cast(tf.shape(images)[1] * tf.shape(images)[2], tf.float32)
    area = tf.cast((bbx2 - bbx1) * (bby2 - bby1), tf.float32)
    lam = 1 - area / img_area

    new_images = tf.identity(images)
    new_images[:, bby1:bby2, bbx1:bbx2, :] = shuffled_images[:, bby1:bby2, bbx1:bbx2, :]

    new_labels = lam * labels + (1 - lam) * shuffled_labels
    return new_images, new_labels

def apply_cutmix(images, labels):
    return cutmix(images, labels)

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image, label

def create_improved_model():
    model = models.Sequential([
        layers.Input(shape=(32,32,3)),
        layers.Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.Activation('gelu'),
        layers.Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.Activation('gelu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3,3), padding='same', kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.Activation('gelu'),
        layers.Conv2D(128, (3,3), padding='same', kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.Activation('gelu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.4),

        layers.Conv2D(256, (3,3), padding='same', kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.Activation('gelu'),
        layers.Conv2D(256, (3,3), padding='same', kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.Activation('gelu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.5),

        layers.Flatten(),
        layers.Dense(256, kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.Activation('gelu'),
        layers.Dropout(0.5),
        layers.Dense(10, dtype='float32'),
    ])

    opt = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def train_improved_model(epochs=110, batch_size=128):
    model = create_improved_model()

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(2048)
    train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_images = test_images[:5000]
    val_labels = test_labels[:5000]

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=18, min_lr=1e-5)
    ]

    history = model.fit(train_dataset, epochs=epochs, validation_data=(val_images, val_labels), callbacks=callbacks)
    return model, history

def evaluate_and_visualize(model, history):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")

    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Val'], loc='lower right')

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Val'], loc='upper right')

    plt.tight_layout()
    plt.show()
    return test_acc

if __name__ == "__main__":
    model, history = train_improved_model()
    evaluate_and_visualize(model, history)