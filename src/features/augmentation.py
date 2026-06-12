import tensorflow as tf

def obter_camada_augmentation(seed):
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
        tf.keras.layers.RandomRotation(0.3, seed=seed),
        tf.keras.layers.RandomZoom(height_factor=(-0.3, -0.1), width_factor=(-0.3, -0.1), seed=seed),
        tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2, seed=seed)
    ], name="aug_destrutivo")