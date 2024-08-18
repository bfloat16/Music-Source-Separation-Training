import numpy as np
import torch
import torch.nn as nn
import yaml
from einops import rearrange, pack, unpack, repeat, reduce
from functools import partial
#from trt_runtime.engine import load_engine
#from trt_runtime.inference import TRTInference
from time import time
from librosa import filters

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]
'''
def get_model_from_trt(model_type, config_path, onnx_model_path):
    global audioprocesser
    global audio_channels
    global trt_inference
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if model_type == 'bs_roformer':
        audio_channels = 2 if config['audio']['stereo'] else 1

    audioprocesser = AudioProcessor_bs(
        None,
        config['audio']['stft_n_fft'],
        config['audio']['stft_hop_length'],
        config['audio']['stft_win_length'],
        config['audio']['stft_normalized']
        )

    shapes = {"stft_repr":((1, 2050, 801, 2), (1, 2050, 801, 2), (1, 2050, 801, 2))}

    engine = load_engine(onnx_model_path, shapes, "fp32", 24)
    trt_inference = TRTInference(engine)
    return engine, config
'''

def get_model(config_path):
    global audioprocesser
    global audio_channels
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config['type'] == 'bs_roformer':
        from models.roformer.bs_roformer import BSRoformer
        audio_channels = 2 if config['audio']['stereo'] else 1
        model = BSRoformer(**config['model'], audio_channels=audio_channels)
        audioprocesser = AudioProcessor_bs(
            None,
            config['audio']['stft_n_fft'],
            config['audio']['stft_hop_length'],
            config['audio']['stft_win_length'],
            config['audio']['stft_normalized']
            )
        
    elif config['type'] == 'mel_band_roformer':
        from models.roformer.mel_band_roformer import MelBandRoformer
        model = MelBandRoformer(
            **config['model'],
            stft_n_fft=config['audio']['stft_n_fft'],
            stft_hop_length=config['audio']['stft_hop_length'],
            stft_win_length=config['audio']['stft_win_length'],
            stft_normalized=config['audio']['stft_normalized'],
            stereo=config['audio']['stereo'],
            )
    else:
        raise ValueError(f"Unknown model type: {config['type']}")
    return model, config

class AudioProcessor_bs:
    def __init__(self, stft_window_fn, stft_n_fft, stft_hop_length, stft_win_length, stft_normalized):
        global audio_channels
        self.audio_channels = audio_channels
        self.stft_window_fn = stft_window_fn

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

    def pre_stft(self, raw_audio):
        device = raw_audio.device

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        # STFT
        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')
        stft_window = self.stft_window_fn(device=device)
        stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        stft_repr = torch.view_as_real(stft_repr)
        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')
        stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')

        return stft_repr

    def pre_istft(self, stft_repr, mask, num_stems):
        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')
        stft_repr = torch.view_as_complex(stft_repr)
        mask = torch.view_as_complex(mask)
        stft_repr = stft_repr * mask

        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)
        stft_window = self.stft_window_fn(device=stft_repr.device)
        recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False)
        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s=self.audio_channels, n=num_stems)

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        return recon_audio
'''
class AudioProcessor_mb():
    def __init__(self, stft_window_fn, stft_n_fft, stft_hop_length, stft_win_length, stft_normalized, stereo, sample_rate, num_bands, num_stems, match_input_audio_length=False):
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized
        )

        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, window=torch.ones(stft_n_fft), return_complex=True).shape[1]

        mel_filter_bank_numpy = filters.mel(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands)

        mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)

        mel_filter_bank[0][0] = 1.

        mel_filter_bank[-1, -1] = 1.

        freqs_per_band = mel_filter_bank > 0

        repeated_freq_indices = repeat(torch.arange(freqs), 'f -> b f', b=num_bands)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            freq_indices = repeat(freq_indices, 'f -> f s', s=2)
            freq_indices = freq_indices * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, 'f s -> (f s)')

        self.freq_indices = freq_indices

        num_bands_per_freq = reduce(freqs_per_band, 'b f -> f', 'sum')

        self.num_bands_per_freq = num_bands_per_freq

        self.match_input_audio_length = match_input_audio_length

    def pre_stft(self, raw_audio):
        device = raw_audio.device

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        batch, channels, raw_audio_length = raw_audio.shape

        istft_length = raw_audio_length if self.match_input_audio_length else None

        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')

        stft_window = self.stft_window_fn(device=device)

        stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')
        stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')

        batch_arange = torch.arange(batch, device=device)[..., None]

        x = stft_repr[batch_arange, self.freq_indices]

        return stft_repr, x, channels, batch, istft_length

    def pre_istft(self, stft_repr, masks, channels, batch, istft_length):
        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

        stft_repr = torch.view_as_complex(stft_repr)
        masks = torch.view_as_complex(masks)

        masks = masks.type(stft_repr.dtype)

        scatter_indices = repeat(self.freq_indices, 'f -> b n f t', b=batch, n=self.num_stems, t=stft_repr.shape[-1])

        stft_repr_expanded_stems = repeat(stft_repr, 'b 1 ... -> b n ...', n=self.num_stems)
        masks_summed = torch.zeros_like(stft_repr_expanded_stems).scatter_add_(2, scatter_indices, masks)

        denom = repeat(self.num_bands_per_freq, 'f -> (f r) 1', r=channels)

        masks_averaged = masks_summed / denom.clamp(min=1e-8)

        stft_repr = stft_repr * masks_averaged

        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        stft_window = self.stft_window_fn(device=stft_repr.device)

        recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False, length=istft_length)

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', b=batch, s=self.audio_channels, n=self.num_stems)

        if self.num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        return recon_audio
'''

def demix_track(config, model, mix, device, progress):
    global audioprocesser
    C = config['audio']['chunk_size']
    N = config['inference']['num_overlap']
    fade_size = C // 10
    step = int(C // N)
    border = C - step
    batch_size = config['inference']['batch_size']

    length_init = mix.shape[-1]

    if length_init > 2 * border and (border > 0):
        mix = nn.functional.pad(mix, (border, border), mode='reflect')

    window_size = C
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window_start = torch.ones(window_size)
    window_middle = torch.ones(window_size)
    window_finish = torch.ones(window_size)
    window_start[-fade_size:] *= fadeout # First audio chunk, no fadein
    window_finish[:fade_size] *= fadein # Last audio chunk, no fadeout
    window_middle[-fade_size:] *= fadeout
    window_middle[:fade_size] *= fadein

    with progress:
        task2 = progress.add_task("Processing", total=mix.shape[1] / step)
        with torch.amp.autocast('cuda'):
            with torch.inference_mode():
                req_shape = (len(config['training']['instruments']),) + tuple(mix.shape)

                result = torch.zeros(req_shape, dtype=torch.float32)
                counter = torch.zeros(req_shape, dtype=torch.float32)
                i = 0
                batch_data = []
                batch_locations = []
                while i < mix.shape[1]:
                    part = mix[:, i:i + C].to(device)
                    length = part.shape[-1]
                    if length < C:
                        if length > C // 2 + 1:
                            part = nn.functional.pad(input=part, pad=(0, C - length), mode='reflect')
                        else:
                            part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                    batch_data.append(part)
                    batch_locations.append((i, length))
                    i += step

                    if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                        arr = torch.stack(batch_data, dim=0)
                        if config['type'] == 'bs_roformer':
                            stft_repr = audioprocesser.pre_stft(arr)
                            mask = model(stft_repr)
                            audio = audioprocesser.pre_istft(stft_repr, mask, 1)
                        elif config['type'] == 'mel_band_roformer':
                            audio = model(arr)

                        window = window_middle
                        if i - step == 0:  # First audio chunk, no fadein
                            window = window_start
                        elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                            window = window_finish

                        for j in range(len(batch_locations)):
                            start, l = batch_locations[j]
                            result[..., start:start+l] += audio[j][..., :l].cpu() * window[..., :l]
                            counter[..., start:start+l] += window[..., :l]

                        batch_data = []
                        batch_locations = []
                    progress.update(task2, advance=1)

                estimated_sources = result / counter
                estimated_sources = estimated_sources.cpu().numpy()
                np.nan_to_num(estimated_sources, copy=False, nan=0.0)

                if length_init > 2 * border and (border > 0):
                    estimated_sources = estimated_sources[..., border:-border]

        return {k: v for k, v in zip(config['training']['instruments'], estimated_sources)}
    
def demix_track_trt(config, model, mix, device, progress):
    global audioprocesser
    global trt_inference
    C = config['audio']['chunk_size']
    N = config['inference']['num_overlap']
    fade_size = C // 10
    step = int(C // N)
    border = C - step
    batch_size = config['inference']['batch_size']

    length_init = mix.shape[-1]

    if length_init > 2 * border and (border > 0):
        mix = nn.functional.pad(mix, (border, border), mode='reflect')

    window_size = C
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window_start = torch.ones(window_size)
    window_middle = torch.ones(window_size)
    window_finish = torch.ones(window_size)
    window_start[-fade_size:] *= fadeout # First audio chunk, no fadein
    window_finish[:fade_size] *= fadein # Last audio chunk, no fadeout
    window_middle[-fade_size:] *= fadeout
    window_middle[:fade_size] *= fadein

    with progress:
        task2 = progress.add_task("Processing", total=mix.shape[1] / step)
        with torch.amp.autocast('cuda'):
            with torch.inference_mode():
                req_shape = (len(config['training']['instruments']),) + tuple(mix.shape)

                result = torch.zeros(req_shape, dtype=torch.float32)
                counter = torch.zeros(req_shape, dtype=torch.float32)
                i = 0
                batch_data = []
                batch_locations = []
                while i < mix.shape[1]:
                    part = mix[:, i:i + C].to(device)
                    length = part.shape[-1]
                    if length < C:
                        if length > C // 2 + 1:
                            part = nn.functional.pad(input=part, pad=(0, C - length), mode='reflect')
                        else:
                            part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                    batch_data.append(part)
                    batch_locations.append((i, length))
                    i += step

                    if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                        arr = torch.stack(batch_data, dim=0)
                        stft_repr = audioprocesser.pre_stft(arr)
                        stft_repr1 = stft_repr.cpu().numpy()
                        print(stft_repr1.shape)
                        time_start = time()
                        output = trt_inference.infer(stft_repr1)
                        time_end = time()
                        print(f"inference done in {time_end - time_start:.3f} seconds")
                        mask = output[0].reshape((1, 2050, 801, 2))
                        print(mask.shape)
                        exit()
                        mask = torch.tensor(mask).to(device)
                        audio = audioprocesser.pre_istft(stft_repr, mask, 1)

                        window = window_middle
                        if i - step == 0:  # First audio chunk, no fadein
                            window = window_start
                        elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                            window = window_finish

                        for j in range(len(batch_locations)):
                            start, l = batch_locations[j]
                            result[..., start:start+l] += audio[j][..., :l].cpu() * window[..., :l]
                            counter[..., start:start+l] += window[..., :l]

                        batch_data = []
                        batch_locations = []
                    progress.update(task2, advance=1)

                estimated_sources = result / counter
                estimated_sources = estimated_sources.cpu().numpy()
                np.nan_to_num(estimated_sources, copy=False, nan=0.0)

                if length_init > 2 * border and (border > 0):
                    estimated_sources = estimated_sources[..., border:-border]

        return {k: v for k, v in zip(config['training']['instruments'], estimated_sources)}