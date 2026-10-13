import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


def _conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False,
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False,
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_unet(in_channels, hr_h, hr_w, base_ch=32):
    """
    U-Net SAME-GRID, EARLY FUSION. `in_channels` s'adapte au nombre de prédicteurs
    (+ orographie) : aucune modification à faire pour passer de 3 à 6 variables.
    Entrée (hr_h, hr_w, in_channels) -> sortie (hr_h, hr_w, 1).
    """
    c = base_ch
    inp = layers.Input((hr_h, hr_w, in_channels), name='input')   # <- early fusion

    s1 = _conv_block(inp, c);    p1 = layers.MaxPooling2D(padding='same')(s1)
    s2 = _conv_block(p1, c * 2); p2 = layers.MaxPooling2D(padding='same')(s2)
    s3 = _conv_block(p2, c * 4); p3 = layers.MaxPooling2D(padding='same')(s3)
    s4 = _conv_block(p3, c * 8); p4 = layers.MaxPooling2D(padding='same')(s4)

    b = _conv_block(p4, c * 16)

    u4 = layers.UpSampling2D(interpolation='bilinear')(b)
    u4 = layers.Resizing(int(s4.shape[1]), int(s4.shape[2]), interpolation='bilinear')(u4)
    u4 = layers.Concatenate()([u4, s4]); u4 = _conv_block(u4, c * 8)

    u3 = layers.UpSampling2D(interpolation='bilinear')(u4)
    u3 = layers.Resizing(int(s3.shape[1]), int(s3.shape[2]), interpolation='bilinear')(u3)
    u3 = layers.Concatenate()([u3, s3]); u3 = _conv_block(u3, c * 4)

    u2 = layers.UpSampling2D(interpolation='bilinear')(u3)
    u2 = layers.Resizing(int(s2.shape[1]), int(s2.shape[2]), interpolation='bilinear')(u2)
    u2 = layers.Concatenate()([u2, s2]); u2 = _conv_block(u2, c * 2)

    u1 = layers.UpSampling2D(interpolation='bilinear')(u2)
    u1 = layers.Resizing(int(s1.shape[1]), int(s1.shape[2]), interpolation='bilinear')(u1)
    u1 = layers.Concatenate()([u1, s1]); u1 = _conv_block(u1, c)

    out = layers.Conv2D(1, 1, activation='linear', name='output')(u1)
    return Model(inputs=inp, outputs=out, name='unet_samegrid_earlyfusion_multivar')


def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))


class RMSE(tf.keras.metrics.Metric):
    def __init__(self, name='rmse', **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_sq = self.add_weight(name='sum_sq', initializer='zeros')
        self.count  = self.add_weight(name='count',  initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.sum_sq.assign_add(tf.reduce_sum(tf.square(y_pred - y_true)))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return tf.sqrt(self.sum_sq / (self.count + 1e-8))

    def reset_state(self):
        self.sum_sq.assign(0.); self.count.assign(0.)


class MAE(tf.keras.metrics.Metric):
    def __init__(self, name='mae', **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_abs = self.add_weight(name='sum_abs', initializer='zeros')
        self.count   = self.add_weight(name='count',   initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.sum_abs.assign_add(tf.reduce_sum(tf.abs(y_pred - y_true)))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.sum_abs / (self.count + 1e-8)

    def reset_state(self):
        self.sum_abs.assign(0.); self.count.assign(0.)
