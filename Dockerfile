FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
# Ada Lovelace (RTX 4050) = sm_89
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV FORCE_CUDA=1

# --- System deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3.8-distutils \
    python3-pip git wget curl vim ca-certificates \
    build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/python3.8 /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8 && \
    pip install --upgrade "pip<24.1" setuptools wheel

# --- PyTorch 2.1 + CUDA 11.8 (Ada-compatible) ---
RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118


# --- BERT deps (fixed: scikit-learn not 'sklearn'; urllib3 unpinned) ---
RUN pip install --no-cache-dir \
    boto3 requests six ipdb \
    h5py html2text nltk progressbar2 \
    onnxruntime \
    scikit-learn urllib3 \
    numpy scipy pandas matplotlib tqdm tensorboard pyyaml \
    transformers datasets

# --- dllogger from NVIDIA (BERT requirement) ---
RUN pip install --no-cache-dir git+https://github.com/NVIDIA/dllogger.git

# Note: nvidia/apex intentionally skipped — brittle to build and only needed
# for mixed-precision training which you can't do at scale on 6GB VRAM anyway.
# If BERT scripts fail without it we'll patch them.

WORKDIR /workspace
CMD ["tail", "-f", "/dev/null"]