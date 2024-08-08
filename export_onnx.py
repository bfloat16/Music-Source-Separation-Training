import torch
import argparse
from utils import get_model_from_config

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", default='bs_roformer', type=str, help="mel_band_roformer, bs_roformer")
parser.add_argument("--config_path", default='./ckpt/model_bs_roformer_ep_317_sdr_12.9755.yaml', type=str, help="path to config file")
parser.add_argument("--start_check_point", default='./ckpt/model_bs_roformer_ep_317_sdr_12.9755.ckpt', type=str, help="Initial checkpoint to valid weights")
parser.add_argument("--input_folder", default='./input', type=str, help="folder with mixtures to process")
parser.add_argument("--store_dir", default='./output', type=str, help="path to store results as wav file")
parser.add_argument("--device_ids", nargs='+', type=int, default=[0], help='list of gpu ids')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True

# 加载模型和配置
model, config = get_model_from_config(args.model_type, args.config_path)

# 加载检查点
state_dict = torch.load(args.start_check_point)
model.load_state_dict(state_dict)

# 检查CUDA是否可用，并将模型移动到相应设备
device = torch.device('cpu')
model = model.to(device)
model.eval()

dummy_input = (torch.randn(1, 2050, 801, 2).to(device))

# 设置导出的ONNX文件路径
onnx_export_path = "model.onnx"

torch.onnx.export(model,
                  dummy_input,
                  onnx_export_path,
                  opset_version=20,
                  input_names=['stft_repr'],
                  output_names=['mask', 'num_stems']
                  )

print(f"Model has been converted to ONNX and saved at {onnx_export_path}")