# 🎨 Stable Diffusion XL Image Generator  

This repository contains a **Gradio-based web application** for generating high-quality AI-generated images using **Stable Diffusion XL**. The app allows users to create stunning images with various artistic styles, adjustable parameters, and both **streaming** and **non-streaming** generation modes.  

## ✨ Features  
- **Stable Diffusion XL (Juggernaut-XL-v9) model** for high-quality image generation  
- **Multiple artistic styles** (Cinematic, Anime, Fantasy, Realistic, etc.)  
- **Streaming mode** to watch the image generation process in real-time  
- **Adjustable settings**: inference steps, guidance scale, aspect ratio, resolution  
- **Negative prompts support** to refine image outputs  
- **Seed control** for reproducibility of images  
- **Upscaling** for higher resolution outputs  

## 🚀 Installation & Setup  
### Prerequisites  
- Python 3.8+  
- CUDA-compatible GPU (for best performance)  
- Dependencies: `torch`, `diffusers`, `gradio`, `opencv-python`, `numpy`, `transformers`, `Pillow`, `accelerate`, `safetensors`

### Steps to Run  
1. Clone the repository:  
   ```bash
   git clone https://github.com/sam-eer31/text-to-image-generation.git
   cd text-to-image-generation
   ```  
2. Install dependencies:  
   ```bash
   start install.bat
   ```
3. Run the application:  
   ```bash
   start launch.bat
   #or
   python txt_to_img.py
   ```  
4. Open the Gradio interface in your browser.  

## 📸 Example Prompts  
- **"A beautiful girl with pink hair"**  
- **"A cute cat wearing a hat"**  
- **"A realistic red la ferrari, cinematic, photorealism"**  

## 🔥 Demo  
<img src="https://github.com/user-attachments/assets/423377e7-cbf8-4481-acd7-722f4ced6d56" width="300">
<img src="https://github.com/user-attachments/assets/fe968721-25f3-4807-ab81-965da24709c2" width="300">
<img src="https://github.com/user-attachments/assets/8636aa3a-1ae1-4f37-89cb-e0baab3ffbd9" width="300">

---

## 🛠 Alternative Model Option

If you prefer to use a different model, you can load the **Realistic Vision V5.1** model instead of the default one. Replace the existing model loading code with the following:

```python
# Load the fine-tuned SD model (Realistic Vision V5.1)
pipeline = StableDiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    torch_dtype=torch.float16
).to("cuda")  # Use GPU
```

*This will allow you to generate more realistic images while maintaining the same workflow. 🚀*


## 📌 Future Enhancements  
- Add more fine-tuning options  
- Implement batch image generation  
- Integrate upscaling with **Real-ESRGAN**  
- Add support for **LoRA** & **ControlNet** models  


## 🔹 Other Details 

The models used in this project were sourced from **Hugging Face**:  

- **Juggernaut XL v9** → [RunDiffusion/Juggernaut-XL-v9](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9)  
- **Realistic Vision V5.1** → [SG161222/Realistic_Vision_V5.1_noVAE](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE)  

---

