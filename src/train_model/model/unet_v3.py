import tensorflow as tf
from tensorflow.keras import layers, Model

def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    
    return x


def attention_gate(x, g, filters):
    x1 = layers.Conv2D(filters, (1, 1), padding='same')(x)
    g1 = layers.Conv2D(filters, (1, 1), padding='same')(g)
    
    out = layers.add([x1, g1])
    out = layers.ReLU()(out)
    out = layers.Conv2D(1, (1, 1), padding='same')(out)
    out = layers.Activation('sigmoid')(out)
    
    return layers.multiply([x, out])

def unet_v3(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Downsampling
    c0 = residual_block(inputs, 32)
    p0 = layers.MaxPooling2D((2, 2))(c0)
    
    c1 = residual_block(p0, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = residual_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = residual_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = residual_block(p3, 512)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = residual_block(p4, 1024)
    
    # Upsampling with attention
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = attention_gate(c4, u6, 512)
    u6 = layers.concatenate([u6, c4])
    c6 = residual_block(u6, 512)
    
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = attention_gate(c3, u7, 256)
    u7 = layers.concatenate([u7, c3])
    c7 = residual_block(u7, 256)
    
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = attention_gate(c2, u8, 128)
    u8 = layers.concatenate([u8, c2])
    c8 = residual_block(u8, 128)
    
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = attention_gate(c1, u9, 64)
    u9 = layers.concatenate([u9, c1])
    c9 = residual_block(u9, 64)
    
    u10 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c9)
    u10 = attention_gate(c0, u10, 32)
    u10 = layers.concatenate([u10, c0])
    c10 = residual_block(u10, 32)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c10)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model
