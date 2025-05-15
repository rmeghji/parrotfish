import os
import numpy as np
import tensorflow as tf
import random
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import combinations, permutations
import soundfile as sf
import librosa
import pywt
from utils.config import Config

config = Config()

def gelu(x):
    """Gaussian Error Linear Unit activation function"""
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

@tf.keras.utils.register_keras_serializable()
class DWTLayer(tf.keras.layers.Layer):
    def __init__(self, wavelet_family='db4', mode='periodization', name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.wavelet_family = wavelet_family
        self.mode = mode
        
        wavelet = pywt.Wavelet(wavelet_family)
        self.dec_lo = tf.constant(wavelet.dec_lo, dtype=tf.float32)
        self.dec_hi = tf.constant(wavelet.dec_hi, dtype=tf.float32)
        self.rec_lo = tf.constant(wavelet.rec_lo, dtype=tf.float32)
        self.rec_hi = tf.constant(wavelet.rec_hi, dtype=tf.float32)
    
    def build(self, input_shape):
        wavelet = pywt.Wavelet(self.wavelet_family)
        self.filter_length = self.dec_lo.shape[0]
        self.channels = input_shape[-1]
        
        self.dec_lo_filter = self.add_weight(
            name='dec_lo',
            shape=(self.filter_length, 1, 1),
            initializer=tf.constant_initializer(wavelet.dec_lo),
            trainable=False
        )
        
        self.dec_hi_filter = self.add_weight(
            name='dec_hi',
            shape=(self.filter_length, 1, 1),
            initializer=tf.constant_initializer(wavelet.dec_hi),
            trainable=False
        )
        
        super().build(input_shape)
    
    def call(self, inputs):
        pad_size = self.filter_length - 1
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_size, pad_size], [0, 0]], mode='REFLECT')
        approx_coeffs = []
        detail_coeffs = []
        
        for c in range(self.channels):
            channel_inputs = padded_inputs[:, :, c:c+1]

            approx = tf.nn.conv1d(
                channel_inputs,
                self.dec_lo_filter,
                stride=2,
                padding='VALID'
            )
            detail = tf.nn.conv1d(
                channel_inputs,
                self.dec_hi_filter,
                stride=2,
                padding='VALID'
            )
            approx = approx[:, (pad_size // 2):-(pad_size // 2) if pad_size > 1 else None, :]
            detail = detail[:, (pad_size // 2):-(pad_size // 2) if pad_size > 1 else None, :]
            
            approx_coeffs.append(approx)
            detail_coeffs.append(detail)
        
        if self.channels > 1:
            approx_coeffs = tf.concat(approx_coeffs, axis=-1)
            detail_coeffs = tf.concat(detail_coeffs, axis=-1)
        else:
            approx_coeffs = approx_coeffs[0]
            detail_coeffs = detail_coeffs[0]
        
        return tf.concat([approx_coeffs, detail_coeffs], axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'wavelet_family': self.wavelet_family,
            'mode': self.mode
        })
        return config

@tf.keras.utils.register_keras_serializable()
class IDWTLayer(tf.keras.layers.Layer):
    def __init__(self, wavelet_family='db4',batch_size=16, mode='periodization', name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.wavelet_family = wavelet_family
        self.mode = mode
        self.batch_size = batch_size   
        
        wavelet = pywt.Wavelet(wavelet_family)
        self.rec_lo = tf.constant(wavelet.rec_lo, dtype=tf.float32)
        self.rec_hi = tf.constant(wavelet.rec_hi, dtype=tf.float32)
    
    def build(self, input_shape):
        self.filter_length = self.rec_lo.shape[0]
        self.in_channels = input_shape[-1] // 2
        wavelet = pywt.Wavelet(self.wavelet_family)
        self.rec_lo_filter = self.add_weight(
            name='rec_lo',
            shape=(self.filter_length, 1, 1),
            initializer=tf.constant_initializer(wavelet.rec_lo),
            trainable=False
        )
        
        self.rec_hi_filter = self.add_weight(
            name='rec_hi',
            shape=(self.filter_length, 1, 1),
            initializer=tf.constant_initializer(wavelet.rec_hi),
            trainable=False
        )
        
        super().build(input_shape)
    
    def call(self, inputs):
        approx_coeffs = inputs[:, :, :self.in_channels]
        detail_coeffs = inputs[:, :, self.in_channels:]

        output_channels = []
        for c in range(self.in_channels):
            approx = approx_coeffs[:, :, c:c+1]
            detail = detail_coeffs[:, :, c:c+1]
            
            approx_up = self._upsample(approx)
            detail_up = self._upsample(detail)
            
            approx_recon = tf.nn.conv1d(
                approx_up,
                self.rec_lo_filter,
                stride=1,
                padding='SAME'
            )
            
            detail_recon = tf.nn.conv1d(
                detail_up,
                self.rec_hi_filter,
                stride=1,
                padding='SAME'
            )
            
            recon = approx_recon + detail_recon
            output_channels.append(recon)
        
        if self.in_channels > 1:
            output = tf.concat(output_channels, axis=-1)
        else:
            output = output_channels[0]
        
        return output

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _upsample(self, x):
        """Vectorized upsampling without loops"""
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        output = tf.zeros([batch_size, seq_len*2, channels], dtype=x.dtype)
        batch_indices = tf.repeat(tf.range(batch_size), seq_len)
        seq_indices = tf.tile(tf.range(0, seq_len*2, 2), [batch_size])
        indices = tf.stack([batch_indices, seq_indices], axis=1)
        updates = tf.reshape(x, [-1, channels])

        return tf.tensor_scatter_nd_update(output, indices, updates)

    
    def get_config(self):
        config = super().get_config()
        config.update({
            'wavelet_family': self.wavelet_family,
            'mode': self.mode
        })
        return config


@tf.keras.utils.register_keras_serializable()
class DownsamplingLayer(tf.keras.layers.Layer):
    def __init__(self, num_filters, filter_size, l1_reg=0.0, l2_reg=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    def build(self, input_shape):
        self.input_channels = input_shape[-1]
        if self.input_channels != self.num_filters:
            self.input_proj = tf.keras.layers.Conv1D(
                self.num_filters,
                1,
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name=f'input_proj_{self.name}'
            )
        else:
            self.input_proj = None
            
        self.conv = tf.keras.layers.Conv1D(
            self.num_filters,
            self.filter_size,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name=f'ds_conv_{self.name}'
        )
        
        self.layer_norm = tf.keras.layers.LayerNormalization(name=f'layer_norm_{self.name}')
        
        super().build(input_shape)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.layer_norm(x)
        x = gelu(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_filters': self.num_filters,
            'filter_size': self.filter_size,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg
        })
        return config


@tf.keras.utils.register_keras_serializable()
class UpsamplingLayer(tf.keras.layers.Layer):
    def __init__(self, num_filters, filter_size, l1_reg=0.0, l2_reg=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    def build(self, input_shape):
        self.input_channels = input_shape[-1]
        if self.input_channels != self.num_filters:
            self.input_proj = tf.keras.layers.Conv1D(
                self.num_filters,
                1,
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name=f'input_proj_{self.name}'
            )
        else:
            self.input_proj = None
        
        half_channels = self.input_channels // 2
        self.approx_conv = tf.keras.layers.Conv1D(
            half_channels,
            self.filter_size,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name=f'approx_conv_{self.name}'
        )
        
        self.detail_conv = tf.keras.layers.Conv1D(
            half_channels,
            self.filter_size,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name=f'detail_conv_{self.name}'
        )
        
        self.approx_gate = tf.keras.layers.Conv1D(
            half_channels,
            1,
            activation='sigmoid',
            padding='same',
            name=f'approx_gate_{self.name}'
        )
        
        self.detail_gate = tf.keras.layers.Conv1D(
            half_channels,
            1,
            activation='sigmoid',
            padding='same',
            name=f'detail_gate_{self.name}'
        )
        
        self.approx_norm = tf.keras.layers.LayerNormalization(name=f'approx_norm_{self.name}')
        self.detail_norm = tf.keras.layers.LayerNormalization(name=f'detail_norm_{self.name}')
        self.output_conv = tf.keras.layers.Conv1D(
            self.num_filters,
            1,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name=f'output_conv_{self.name}'
        )
        self.output_norm = tf.keras.layers.LayerNormalization(name=f'output_norm_{self.name}')
        
        super().build(input_shape)

    def call(self, inputs):
        half_channels = self.input_channels // 2
        approx_coeff = inputs[:, :, :half_channels]
        detail_coeff = inputs[:, :, half_channels:]
        
        approx_processed = self.approx_conv(approx_coeff)
        approx_processed = self.approx_norm(approx_processed)
        approx_processed = gelu(approx_processed)
        
        detail_processed = self.detail_conv(detail_coeff)
        detail_processed = self.detail_norm(detail_processed)
        detail_processed = gelu(detail_processed)
        
        approx_gate = self.approx_gate(approx_processed)
        detail_gate = self.detail_gate(detail_processed)
        approx_gated = approx_processed * approx_gate
        detail_gated = detail_processed * detail_gate
        
        combined = tf.concat([approx_gated, detail_gated], axis=-1)
        output = self.output_conv(combined)
        output = self.output_norm(output)
        output = gelu(output)
        
        return output 
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_filters': self.num_filters,
            'filter_size': self.filter_size,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg
        })
        return config

@tf.keras.utils.register_keras_serializable()
class GatedSkipConnection(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        decoder_shape, encoder_shape = input_shape
        self.decoder_channels = decoder_shape[-1]
        self.encoder_channels = encoder_shape[-1]
        
        self.decoder_gate = tf.keras.layers.Conv1D(
            self.decoder_channels,
            1,
            activation='sigmoid',
            padding='same',
            name=f'decoder_gate_{self.name}'
        )
        
        self.encoder_gate = tf.keras.layers.Conv1D(
            self.encoder_channels,
            1,
            activation='sigmoid',
            padding='same',
            name=f'encoder_gate_{self.name}'
        )
        
        self.norm = tf.keras.layers.LayerNormalization(name=f'skip_norm_{self.name}')
        
        super().build(input_shape)
        
    def call(self, inputs):
        decoder_features, encoder_features = inputs
        decoder_gated = decoder_features * self.decoder_gate(decoder_features)
        encoder_gated = encoder_features * self.encoder_gate(encoder_features)
        concat = tf.concat([decoder_gated, encoder_gated], axis=-1)
        return self.norm(concat)
    
    def get_config(self):
        config = super().get_config()
        return config

@tf.keras.utils.register_keras_serializable()
class WaveletUNet(tf.keras.Model):
    def __init__(self, num_coeffs, wavelet_depth, batch_size, channels, num_layers, 
                 num_init_filters, filter_size, merge_filter_size, l1_reg, l2_reg,
                 max_sources=3, wavelet_family='db4', **kwargs):
        super().__init__(**kwargs)
        self.num_coeffs = num_coeffs
        self.wavelet_depth = wavelet_depth + 1
        self.batch_size = batch_size
        self.channels = channels
        self.num_layers = num_layers
        self.num_init_filters = num_init_filters
        self.filter_size = filter_size
        self.merge_filter_size = merge_filter_size
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.max_sources = max_sources
        self.wavelet_family = wavelet_family

    @classmethod
    def from_config(cls, config):
        num_coeffs = config.pop('num_coeffs')
        wavelet_depth = config.pop('wavelet_depth')
        batch_size = config.pop('batch_size')
        channels = config.pop('channels')
        num_layers = config.pop('num_layers')
        num_init_filters = config.pop('num_init_filters')
        filter_size = config.pop('filter_size')
        merge_filter_size = config.pop('merge_filter_size')
        l1_reg = config.pop('l1_reg')
        l2_reg = config.pop('l2_reg')
        max_sources = config.pop('max_sources', 3)
        wavelet_family = config.pop('wavelet_family', 'db4')

        return cls(
            num_coeffs=num_coeffs,
            wavelet_depth=wavelet_depth,
            batch_size=batch_size,
            channels=channels,
            num_layers=num_layers,
            num_init_filters=num_init_filters,
            filter_size=filter_size,
            merge_filter_size=merge_filter_size,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            max_sources=max_sources,
            wavelet_family=wavelet_family,
            **config
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_coeffs': self.num_coeffs,
            'wavelet_depth': self.wavelet_depth,
            'batch_size': self.batch_size,
            'channels': self.channels,
            'num_layers': self.num_layers,
            'num_init_filters': self.num_init_filters,
            'filter_size': self.filter_size,
            'merge_filter_size': self.merge_filter_size,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'max_sources': self.max_sources,
            'wavelet_family': self.wavelet_family
        })
        return config

    def build(self, input_shape):
        self.initial_conv = tf.keras.layers.Conv1D(
            self.num_init_filters,
            self.filter_size,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name='initial_conv'
        )
        self.initial_norm = tf.keras.layers.LayerNormalization(name='initial_norm')
        self.downsampling_blocks = {}
        self.dwt_layers = {}
        self.down_process_blocks = {}
        
        for i in range(self.num_layers):
            block_name = f'{i+1}'
            num_filters = self.num_init_filters + (self.num_init_filters * i)
            
            self.downsampling_blocks[block_name] = DownsamplingLayer(
                num_filters, 
                self.filter_size, 
                l1_reg=self.l1_reg, 
                l2_reg=self.l2_reg,
                name=f'ds_block_{block_name}'
            )
            
            self.dwt_layers[block_name] = DWTLayer(
                wavelet_family=self.wavelet_family,
                name=f'dwt_{block_name}'
            )
            
            self.down_process_blocks[block_name] = UpsamplingLayer(
                num_filters * 2,
                self.filter_size,
                l1_reg=self.l1_reg,
                l2_reg=self.l2_reg,
                name=f'down_process_{block_name}'
            )

        self.bottle_neck = tf.keras.Sequential([
            tf.keras.layers.Conv1D(
                self.num_init_filters * (self.num_layers + 1),
                self.filter_size,
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name='bottleneck_conv1'
            ),
            tf.keras.layers.LayerNormalization(name='bottleneck_norm2'),
            tf.keras.layers.Lambda(gelu)
        ])

        self.idwt_layers = {}
        self.up_process_blocks = {}
        self.skip_connections = {}
        self.upsampling_blocks = {}
        
        for i in range(self.num_layers):
            block_name = f'{self.num_layers - i}'
            num_filters = self.num_init_filters + (self.num_init_filters * (self.num_layers - i - 1))
            
            self.idwt_layers[block_name] = IDWTLayer(
                wavelet_family=self.wavelet_family,
                name=f'idwt_{block_name}'
            )
            
            self.up_process_blocks[block_name] = UpsamplingLayer(
                num_filters,
                self.filter_size,
                l1_reg=self.l1_reg,
                l2_reg=self.l2_reg,
                name=f'up_process_{block_name}'
            )
            
            self.skip_connections[block_name] = GatedSkipConnection(
                name=f'skip_connection_{block_name}'
            )
            
            self.upsampling_blocks[block_name] = tf.keras.Sequential([
                tf.keras.layers.Conv1D(
                    num_filters,
                    self.merge_filter_size,
                    padding='same',
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                    name=f'up_conv1_{block_name}'
                ),
                tf.keras.layers.LayerNormalization(name=f'up_norm1_{block_name}'),
                tf.keras.layers.Lambda(gelu),
                tf.keras.layers.Conv1D(
                    num_filters,
                    self.merge_filter_size,
                    padding='same',
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                    name=f'up_conv2_{block_name}'
                ),
                tf.keras.layers.LayerNormalization(name=f'up_norm2_{block_name}'),
                tf.keras.layers.Lambda(gelu)
            ])

        self.final_conv = tf.keras.Sequential([
            tf.keras.layers.Conv1D(
                self.num_init_filters,
                self.filter_size,
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name='final_conv1'
            ),
            tf.keras.layers.LayerNormalization(name='final_norm1'),
            tf.keras.layers.Lambda(gelu),
            tf.keras.layers.Conv1D(
                self.num_init_filters,
                self.filter_size,
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name='final_conv2'
            ),
            tf.keras.layers.LayerNormalization(name='final_norm2'),
            tf.keras.layers.Lambda(gelu)
        ])

        # uncomment for residual version (pre 5-14-25)
        self.output_conv = tf.keras.layers.Conv1D(
            self.max_sources - 1,
            1,
            activation='tanh',
            padding='same',
            name='output_conv'
        )

        # uncomment for non residual version (post 5-14-25)
        # self.output_conv = tf.keras.layers.Conv1D(
        #     self.max_sources ,
        #     1,
        #     activation='tanh',
        #     padding='same',
        #     name='output_conv'
        # )
            
        super().build(input_shape)
        self.summary()

    def call(self, inputs, training=True):
        current_layer = self.initial_conv(inputs)
        current_layer = self.initial_norm(current_layer)
        current_layer = gelu(current_layer)
        full_mix = tf.reduce_sum(inputs, axis=-1, keepdims=True)

        enc_outputs = {}
        for i in range(self.num_layers):
            block_name = f'{i+1}'
            current_layer = self.downsampling_blocks[block_name](current_layer)
            enc_outputs[block_name] = current_layer
            current_layer = self.dwt_layers[block_name](current_layer)
            current_layer = self.down_process_blocks[block_name](current_layer)

        current_layer = self.bottle_neck(current_layer)

        for i in range(self.num_layers):
            block_name = f'{self.num_layers - i}'
            current_layer = self.idwt_layers[block_name](current_layer)
            current_layer = self.up_process_blocks[block_name](current_layer)
            skip_conn = enc_outputs[block_name]
            
            if current_layer.shape[1] != skip_conn.shape[1]:
                diff = skip_conn.shape[1] - current_layer.shape[1]
                if diff > 0:
                    pad_start = diff // 2
                    pad_end = diff - pad_start
                    current_layer = tf.pad(current_layer, [[0, 0], [pad_start, pad_end], [0, 0]], mode='SYMMETRIC')
                else:
                    diff = -diff
                    crop_start = diff // 2
                    current_layer = tf.slice(current_layer, [0, crop_start, 0], [-1, skip_conn.shape[1], -1])
            current_layer = self.skip_connections[block_name]([current_layer, skip_conn])
            current_layer = self.upsampling_blocks[block_name](current_layer)

        current_layer = self.final_conv(current_layer)

        if current_layer.shape[1] != self.num_coeffs:
            diff = self.num_coeffs - current_layer.shape[1]
            if diff > 0:
                pad_start = diff // 2
                pad_end = diff - pad_start
                current_layer = tf.pad(current_layer, [[0, 0], [pad_start, pad_end], [0, 0]], mode='SYMMETRIC')
            else:
                diff = -diff
                crop_start = diff // 2
                current_layer = tf.slice(current_layer, [0, crop_start, 0], [-1, self.num_coeffs, -1])

        current_layer = tf.concat([full_mix, current_layer], axis=-1)
        
        # uncomment for residual version (pre 5-14-25)
        partial_outputs = self.output_conv(current_layer)
        residual_source = full_mix - tf.reduce_sum(partial_outputs, axis=-1, keepdims=True)
        outputs = tf.concat([partial_outputs, residual_source], axis=-1)
        
        # uncomment for non residual version (post 5-14-25)
        # outputs = self.output_conv(current_layer)
        
        outputs = tf.transpose(outputs, [0, 2, 1])  # (batch, sources, time)
        
        return outputs

@tf.keras.utils.register_keras_serializable()
def pit_loss(y_true, y_pred):
    """
    Permutation Invariant Training loss for audio source separation with three sources.
    
    Args:
        y_true: Ground truth with shape [batch, features, 3]
               Sources with fewer than 3 speakers are zero-padded.
        y_pred: Model prediction with shape [batch, features, 3]
               
    Returns:
        The PIT loss value (scalar)
    """
    y_true_s1 = y_true[:, 0, :]
    y_true_s2 = y_true[:, 1, :]
    y_true_s3 = y_true[:, 2, :]
    
    y_pred_s1 = y_pred[:, 0, :]
    y_pred_s2 = y_pred[:, 1, :]
    y_pred_s3 = y_pred[:, 2, :]
    
    mse_1_1 = tf.reduce_mean(tf.square(y_true_s1 - y_pred_s1), axis=1)
    mse_1_2 = tf.reduce_mean(tf.square(y_true_s1 - y_pred_s2), axis=1)
    mse_1_3 = tf.reduce_mean(tf.square(y_true_s1 - y_pred_s3), axis=1)
    mse_2_1 = tf.reduce_mean(tf.square(y_true_s2 - y_pred_s1), axis=1)
    mse_2_2 = tf.reduce_mean(tf.square(y_true_s2 - y_pred_s2), axis=1)
    mse_2_3 = tf.reduce_mean(tf.square(y_true_s2 - y_pred_s3), axis=1)
    mse_3_1 = tf.reduce_mean(tf.square(y_true_s3 - y_pred_s1), axis=1)
    mse_3_2 = tf.reduce_mean(tf.square(y_true_s3 - y_pred_s2), axis=1)
    mse_3_3 = tf.reduce_mean(tf.square(y_true_s3 - y_pred_s3), axis=1)
    
    loss_perm1 = (mse_1_1 + mse_2_2 + mse_3_3) / 3.0
    loss_perm2 = (mse_1_1 + mse_2_3 + mse_3_2) / 3.0
    loss_perm3 = (mse_1_2 + mse_2_1 + mse_3_3) / 3.0
    loss_perm4 = (mse_1_2 + mse_2_3 + mse_3_1) / 3.0
    loss_perm5 = (mse_1_3 + mse_2_1 + mse_3_2) / 3.0
    loss_perm6 = (mse_1_3 + mse_2_2 + mse_3_1) / 3.0
    all_losses = tf.stack([loss_perm1, loss_perm2, loss_perm3, 
                          loss_perm4, loss_perm5, loss_perm6], axis=1)
    min_loss = tf.reduce_min(all_losses, axis=1)
    
    return tf.reduce_mean(min_loss)