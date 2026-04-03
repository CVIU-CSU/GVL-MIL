cd /root/userfolder/MIL/VL-MIL
pip install --upgrade pip
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[train]"
pip install flash-attn==2.7.3 --no-build-isolation