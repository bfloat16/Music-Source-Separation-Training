import os
import glob
import torch
import librosa
import argparse
import numpy as np
import soundfile as sf
import torch.nn as nn
from utils import demix_track, demix_track_trt, get_model
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
progress = Progress(TextColumn("Running: "), BarColumn(), "[progress.percentage]{task.percentage:>3.1f}%", "•", MofNCompleteColumn(), "•", TimeElapsedColumn(), "|", TimeRemainingColumn())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default='ckpt/karaoke_mel_band_roformer.yaml', type=str, help="path to config file")
    parser.add_argument("--start_check_point", default='ckpt/karaoke_mel_band_roformer.ckpt', type=str, help="Initial checkpoint to valid weights")
    parser.add_argument("--onnx", default='bsr.onnx', type=str, help="path to trt model")
    parser.add_argument("--input_folder", default='input', type=str, help="folder with mixtures to process")
    parser.add_argument("--store_dir", default='output', type=str, help="path to store results as wav file")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    args = parser.parse_args()
    return args

def run_folder(model, args, config, device):
    model.eval()
    all_mixtures_path = glob.glob(os.path.join(args.input_folder, '**', '*.*'), recursive=True)
    print('Total files found: {}'.format(len(all_mixtures_path)))

    instruments = config['training']['instruments']

    with progress:
        task1 = progress.add_task("Total", total=len(all_mixtures_path))
        for path in all_mixtures_path:
            try:
                mix, sr = librosa.load(path, sr=44100, mono=False)
                mix = mix.T
            except Exception as e:
                print(f'Cannot read track: {path}')
                print(f'Error message: {e}')
                continue

            if mix.ndim == 1:
                mix = np.stack([mix, mix], axis=-1)

            mixture = torch.tensor(mix.T, dtype=torch.float32).to(device)
            res = demix_track(config, model, mixture, device, progress)

            progress.update(task1, advance=1)

            filename = os.path.splitext(os.path.basename(path))[0]
            relative_path = os.path.relpath(path, args.input_folder)
            output_dir = os.path.join(args.store_dir, os.path.dirname(relative_path))
            os.makedirs(output_dir, exist_ok=True)

            for instr in instruments:
                vocal_output_file = os.path.join(output_dir, f"{filename}_vocals.wav")
                instrumental_output_file = os.path.join(output_dir, f"{filename}_instrum.wav")

                sf.write(vocal_output_file, res[instr].T, sr, subtype='PCM_16')
                sf.write(instrumental_output_file, (mix - res[instr].T), sr, subtype='PCM_16')

def run_folder_trt(model, args, config, device):
    all_mixtures_path = glob.glob(os.path.join(args.input_folder, '**', '*.*'), recursive=True)
    print('Total files found: {}'.format(len(all_mixtures_path)))

    instruments = config['training']['instruments']

    with progress:
        task1 = progress.add_task("Total", total=len(all_mixtures_path))
        for path in all_mixtures_path:
            try:
                mix, sr = librosa.load(path, sr=44100, mono=False)
                mix = mix.T
            except Exception as e:
                print(f'Cannot read track: {path}')
                print(f'Error message: {e}')
                continue

            if mix.ndim == 1:
                mix = np.stack([mix, mix], axis=-1)

            mixture = torch.tensor(mix.T, dtype=torch.float32).to(device)
            res = demix_track_trt(config, model, mixture, device, progress)

            progress.update(task1, advance=1)

            filename = os.path.splitext(os.path.basename(path))[0]
            relative_path = os.path.relpath(path, args.input_folder)
            output_dir = os.path.join(args.store_dir, os.path.dirname(relative_path))
            os.makedirs(output_dir, exist_ok=True)

            for instr in instruments:
                vocal_output_file = os.path.join(output_dir, f"{filename}_vocals.wav")
                instrumental_output_file = os.path.join(output_dir, f"{filename}_instrum.wav")

                sf.write(vocal_output_file, res[instr].T, sr, subtype='PCM_16')
                sf.write(instrumental_output_file, (mix - res[instr].T), sr, subtype='PCM_16')

def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    model, config = get_model(args.config_path)
    if args.start_check_point != '':
        print('Start from checkpoint: {}'.format(args.start_check_point))
        state_dict = torch.load(args.start_check_point, weights_only=True)
        model.load_state_dict(state_dict)
    print("Instruments: {}".format(config['training']['instruments']))

    if torch.cuda.is_available():
        device_ids = args.device_ids
        if type(device_ids)==int:
            device = torch.device(f'cuda:{device_ids}')
            model = model.to(device)
        else:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = 'cpu'
        print('CUDA is not avilable. Run inference on CPU. It will be very slow...')
        model = model.to(device)

    run_folder(model, args, config, device)
'''
def main_trt():
    args = parse_args()
    engine, config = get_model_from_trt(args.model_type, args.config_path, args.onnx)
    run_folder_trt(engine, args, config, 'cuda')
'''

if __name__ == "__main__":
    main()
    #main_trt()