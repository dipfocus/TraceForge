# Install PyTorch (CUDA 12.8)
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/EasternJournalist/utils3d.git#egg=utils3d

pip install kornia==0.8.1 huggingface_hub
pip install hydra-core omegaconf
pip install timm PyJWT gdown rich
pip install "ray[default]"
pip install jaxtyping tqdm

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install av typed-argument-parser tqdm scipy h5py kornia
pip install sophuspy trimesh
pip install "python-box[all]~=7.0"
pip install wandb loguru
pip install opencv-python einops matplotlib Pillow pyrealsense2
pip install viser mediapy
pip install scikit-learn

# For text generation (VLM instruction generation)
pip install python-dotenv openai google-genai

cd ./third_party/pointops2
python setup.py install
pip install flow_vis moviepy==1.0.0 easydict
