import numpy as np
import tensorflow as tf
import librosa
import cv2
import tqdm
import pywt
import os

class WaveData:
    def __init__(self,
                 identifier: str,
                 waveform: np.ndarray, 
                 dwt: np.ndarray = None,
                 speaker_id: str = None,
                 is_mixed: bool = False,
                 source_ids: list = None) -> None:
        
        self.identifier = identifier
        self.waveform = waveform
        self.dwt = dwt
        self.speaker_id = speaker_id
        self.is_mixed = is_mixed
        self.source_ids = source_ids
        self.tensor_coeffs = None

    def add_source(self, source_id: str) -> None:
        """Add a source ID to this mixed track"""
        if self.source_ids is None:
            self.source_ids = []

        if source_id not in self.source_ids:
            self.source_ids.append(source_id)

    def get_metadata(self):
        """Return metadata about this audio sample"""
        return {
            "identifier": self.identifier,
            "speaker_id": self.speaker_id,
            "is_mixed": self.is_mixed,
            "num_sources": len(self.source_ids),
            "source_ids": self.source_ids,
            "has_dwt": self.dwt is not None,
            "has_tensor_coeffs": self.tensor_coeffs is not None,
            "waveform_shape": self.waveform.shape if self.waveform is not None else None
        }
        
def makeWaveDictBatch(mixed_waveforms: list, all_source_waveforms: list, all_source_ids: list, mixed_ids: list) -> dict:
    '''
    Make a dictionary of wavelet data

    params:
    - mixed_waveforms: list, list of mixed waveforms with shape (num_mixed, num_samples)
    - all_source_waveforms: list, list of source waveforms with shape (num_mixed, num_sources, num_samples)
    - all_source_ids: list, list containing list of source ids for each mixed waveform
    - mixed_ids: list, list of mixed ids

    return: 
    - data: dict, dictionary of wavelet data
    '''
    data = {}
    for i, (mixed_waveform, source_waveforms, source_ids) in enumerate(tqdm.tqdm(zip(mixed_waveforms, all_source_waveforms, all_source_ids), desc=f'Loading waveforms', total=len(mixed_waveforms), leave=False)):
        mix_id = mixed_ids[i]
        
        data[mix_id] = WaveData(identifier=mix_id, waveform=mixed_waveform, dwt=None, is_mixed=True, source_ids=source_ids)
        for j, source_id in enumerate(source_ids):
            if source_id not in data: # add source to dict if not already there
                data[source_id] = WaveData(identifier=source_id, waveform=source_waveforms[j], dwt=None, is_mixed=False)
            if source_id not in data[mix_id].source_ids: # add source id to mix sources metadata if not already there
                data[mix_id].add_source(source_id)

    return data

def getWaveletTransform(data: dict, song: str, level: int=5) -> dict:
    '''
    Get the wavelet transform of the waveform

    params:
    - data: dict, dictionary of wavelet data
    - song: str, song key
    - level: int, level of wavelet decomposition

    return: 
    - data: dict, updated dictionary of wavelet data
    '''

    # ensure the waveform is in the correct shape
    if data[song].waveform.shape[0] == 2 or data[song].waveform.shape[0] == 1:
        data[song].waveform = np.transpose(data[song].waveform)
    # print(f"Left channel waveform: {data[song].waveform[:, 0]}")

    # Perform wavelet decomposition
    coeffs_left = pywt.wavedec(data[song].waveform[:, 0], 'haar', level=level, mode='symmetric')

    
    # Find the maximum length among all coefficients
    # max_len = max([c.shape[0] for c in coeffs_left + coeffs_right])
    max_len = max([c.shape[0] for c in coeffs_left])

    # print([c.shape[0] for c in coeffs_left])

    # Stretch the coefficients to the maximum length using interpolation
    stretched_coeffs_left = []
    # stretched_coeffs_right = []
    for c_left in coeffs_left:
        stretched_left = cv2.resize(c_left.reshape(1, -1), (max_len, 1), interpolation=cv2.INTER_NEAREST).flatten()
        stretched_coeffs_left.append(stretched_left)

    ## transpose
    stretched_coeffs_left = np.transpose(stretched_coeffs_left)
    tensor_coeffs = tf.convert_to_tensor(stretched_coeffs_left)

    # Update the data object
    data[song].dwt = coeffs_left
    data[song].tensor_coeffs = tensor_coeffs

    return data