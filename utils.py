import numpy as np
import torch
import torch.nn as nn
import yaml

def get_model_from_config(model_type, config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if model_type == 'bs_roformer':
        from models.bs_roformer import BSRoformer
        model = BSRoformer(**config['model'])
    else:
        print('Unknown model: {}'.format(model_type))
        model = None
    return model, config

def demix_track(config, model, mix, device, progress):
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
                        x = model(arr)

                        window = window_middle
                        if i - step == 0:  # First audio chunk, no fadein
                            window = window_start
                        elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                            window = window_finish

                        for j in range(len(batch_locations)):
                            start, l = batch_locations[j]
                            result[..., start:start+l] += x[j][..., :l].cpu() * window[..., :l]
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

def demix_track_demucs(config, model, mix, device):
    S = len(config.training.instruments)
    C = config.training.samplerate * config.training.segment
    N = config.inference.num_overlap
    batch_size = config.inference.batch_size
    step = C // N

    with torch.cuda.amp.autocast(enabled=config.training.use_amp):
        with torch.inference_mode():
            req_shape = (S, ) + tuple(mix.shape)
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)
            i = 0
            batch_data = []
            batch_locations = []
            while i < mix.shape[1]:
                # print(i, i + C, mix.shape[1])
                part = mix[:, i:i + C].to(device)
                length = part.shape[-1]
                if length < C:
                    part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                batch_data.append(part)
                batch_locations.append((i, length))
                i += step

                if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)
                    for j in range(len(batch_locations)):
                        start, l = batch_locations[j]
                        result[..., start:start+l] += x[j][..., :l].cpu()
                        counter[..., start:start+l] += 1.
                    batch_data = []
                    batch_locations = []

            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

    if S > 1:
        return {k: v for k, v in zip(config.training.instruments, estimated_sources)}
    else:
        return estimated_sources

def sdr(references, estimates):
    # compute SDR for one song
    delta = 1e-7  # avoid numerical errors
    num = np.sum(np.square(references), axis=(1, 2))
    den = np.sum(np.square(references - estimates), axis=(1, 2))
    num += delta
    den += delta
    return 10 * np.log10(num / den)