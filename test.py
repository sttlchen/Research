import torch, platform, sys, os
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("arch:", platform.machine(), "| python:", sys.version)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("cudnn:", torch.backends.cudnn.version())
