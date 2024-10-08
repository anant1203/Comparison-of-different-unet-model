import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

# Define the base model with MobileNetV2
def create_down_stack():
    base_model = tf.keras.applications.MobileNetV2(include_top=False)

    # Layers from MobileNetV2 to extract features
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False  # Freeze the weights of the pre-trained model

    return down_stack

# Define the upsampling stack
def create_up_stack():
    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]
    return up_stack

# Mobilenet segmentation model function
def mobilenet_model(output_channels:int):
    down_stack = create_down_stack()
    up_stack = create_up_stack()

    inputs = tf.keras.layers.Input(shape=[224, 224, 3])

    # Downsampling
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling with skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # Output layer (this example uses Conv2DTranspose for upsampling)
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2, padding='same')  # 64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)