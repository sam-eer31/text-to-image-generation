@echo off

echo "Creating a Virtual Environment..."
call python -m venv myenv

echo "Activating the Virtual Environment..."
call myenv\Scripts\activate

echo "Installing dependencies..."
call pip install diffusers transformers gradio numpy opencv-python Pillow accelerate safetensors

echo "Installing PyTorch with CUDA 12.4..."
call pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

echo "Installation complete!"
pause