FROM nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3

RUN ["apt-get", "update"]
RUN ["apt", "-y", "install", "python3-pip"]
RUN ["apt", "-y", "install", "python-setuptools"]
RUN ["apt-get", "-y", "install", "python3-opencv"]
RUN ["apt-get", "-y", "install", "protobuf-compiler", "libprotoc-dev"]

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt --trusted-host pypi.python.org --trusted-host pypi.org --trusted-host files.pythonhosted.org

### Install Bazel (required to compile torch_tensorrt)
COPY install_bazel.sh .
RUN ["bash", "install_bazel.sh"]
RUN ["ln", "/root/src/bazel-3.1.0-dist/output/bazel", "/usr/bin/bazel"]

### Install pytorch 1.10.0 (required to compile torch_tensorrt)
RUN ["wget", "https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl", "-O", "torch-1.10.0-cp36-cp36m-linux_aarch64.whl"]
RUN ["pip3", "install", "numpy", "torch-1.10.0-cp36-cp36m-linux_aarch64.whl"]

### Compile tensorrt
RUN ["git", "clone", "https://github.com/NVIDIA/Torch-TensorRT.git"]
WORKDIR /Torch-TensorRT
RUN ["git", "checkout", "9ddd7a883934322e1b0f7ca0094ca597f5517174"]
WORKDIR /
COPY WORKSPACE Torch-TensorRT
WORKDIR "/Torch-TensorRT/py"
RUN ["python3", "setup.py", "install", "--use-cxx11-abi"]

### Install torchvision 0.11.1 (compatible with pytorch 1.10.0)
WORKDIR /YOLOv5-PyTorch
RUN ["apt-get", "install", "libjpeg-dev", "zlib1g-dev"]
RUN ["git", "clone", "-b", "v0.11.1", "https://github.com/pytorch/vision", "torchvision"]
WORKDIR torchvision
RUN ["python3", "setup.py", "install"]

### Run tests
WORKDIR /YOLOv5-PyTorch
COPY . .
RUN ["python3", "test.py"]
