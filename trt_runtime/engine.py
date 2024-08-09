import os
import hashlib
import tensorrt as trt
from cuda import cudart

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def calculate_hash(onnx_model_path, shapes, precision, workspace_size, trt_version, device_name):
    hash_input = f"{onnx_model_path}_{shapes}_{precision}_{workspace_size}_{trt_version}_{device_name}"
    return hashlib.sha256(hash_input.encode()).hexdigest()

def build_engine(onnx_model_path, shapes, precision, workspace_size):
    # Check precision
    if precision not in ['fp32', 'fp16', 'tf32']:
        raise ValueError(f"Precision must be 'fp32', 'fp16', or 'tf32', got {precision}")

    # Load the ONNX model
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

    # Calculate current environment hash
    trt_version = trt.__version__
    device_name = cudart.cudaGetDeviceProperties(0)[1].name.decode('utf-8')
    current_hash = calculate_hash(onnx_model_path, shapes, precision, workspace_size, trt_version, device_name)

    # Check if engine already exists and is up to date
    engine_path = os.path.splitext(onnx_model_path)[0] + ".engine"
    sig_path = os.path.splitext(onnx_model_path)[0] + ".sig"
    
    if os.path.exists(engine_path) and os.path.exists(sig_path):
        with open(sig_path, 'r') as hash_file:
            saved_hash = hash_file.read()
        if saved_hash == current_hash:
            print("Engine is up to date")
            return engine_path
        else:
            print("Engine is outdated, rebuilding...")
    else:
        print("Engine does not exist, building...")

    # Create a TRT logger and builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_model_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            raise RuntimeError("Failed to parse the ONNX model")

    # Create builder config
    config = builder.create_builder_config()
    max_workspace_size = int(workspace_size) << 30
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    
    if precision == 'fp16' and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'tf32':
        config.set_flag(trt.BuilderFlag.TF32)
    
    # Create optimization profile
    profile = builder.create_optimization_profile()
    
    for input_name, (min_shape, opt_shape, max_shape) in shapes.items():
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    
    config.add_optimization_profile(profile)

    # Build the engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build the engine")

    # Serialize the engine and save it
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    # Save the current hash
    with open(sig_path, 'w') as hash_file:
        hash_file.write(current_hash)

    return engine_path

def load_engine(onnx_model_path, shapes, precision, workspace_size):
    engine_path = build_engine(onnx_model_path, shapes, precision, workspace_size)
    
    runtime = trt.Runtime(TRT_LOGGER)
    
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    return engine