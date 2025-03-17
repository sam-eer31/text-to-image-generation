import os
# Set environment variables to suppress warnings and improve performance
os.environ["XFORMERS_IGNORE_MISMATCH"] = "1"  # Ignore version mismatches in xformers
os.environ["XFORMERS_MORE_DETAILS"] = "0"  # Reduce verbosity of xformers
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Disable symlink warnings from Hugging Face Hub

import torch
import gradio as gr
from PIL import Image
import numpy as np
import cv2
import threading
import queue
from diffusers import DiffusionPipeline  # Import the DiffusionPipeline for Stable Diffusion

# Load the Stable Diffusion model
try:
    # Load the Juggernaut-XL-v9 model with specific configurations
    pipe = DiffusionPipeline.from_pretrained(
        "RunDiffusion/Juggernaut-XL-v9",
        torch_dtype=torch.float16,  # Use half-precision for faster inference
        variant="fp16",  # Specify the floating-point precision
        use_safetensors=True  # Use safetensors for safer model loading
    ).to("cuda")  # Move the model to GPU for faster processing

    print("Stable Diffusion model loaded successfully.")

except Exception as e:
    print(f"Error loading Stable Diffusion model: {e}")
    exit()  # Exit the program if model loading fails

# Define detailed style prompts for different artistic styles
STYLE_PROMPTS = {
    "none": "",  # No additional style
    "cinematic": "cinematic still, highly detailed, dramatic lighting, film grain, 8k, ultra-realistic, depth of field, bokeh",
    "cartoon": "cartoon style, vibrant colors, bold outlines, whimsical, animated, Pixar-like, playful, 2D animation",
    "model": "studio lighting, elegant, photorealistic, Vogue magazine style, professional photoshoot",
    "realistic": "ultra-realistic, photorealistic, highly detailed, 8k, natural lighting, sharp focus, lifelike textures",
    "fantasy": "fantasy art, magical, ethereal, otherworldly, intricate details, glowing elements, mystical atmosphere, concept art",
    "anime": "anime style, vibrant colors, expressive eyes, stylized, Japanese animation, Studio Ghibli-inspired, cel-shaded"
}

# Define available styles as a list of keys from STYLE_PROMPTS
STYLES = list(STYLE_PROMPTS.keys())

# Define a default negative prompt to avoid undesirable image characteristics
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, overexposed, underexposed, poorly lit, bad anatomy, extra limbs, disfigured, deformed, out of frame"

# Define additional style-specific negative prompts to refine the output for each style
STYLE_NEGATIVE_PROMPTS = {
    "none": "",  # No additional negative prompt
    "cinematic": "flat, dull, low contrast, unrealistic, fake, CGI",
    "cartoon": "realistic, photorealistic, 3D, dull, flat",
    "model": "cartoonish, animated, flat, unrealistic, fake",
    "realistic": "cartoonish, animated, flat, unrealistic, fake",
    "fantasy": "realistic, photorealistic, flat, dull, boring",
    "anime": "realistic, photorealistic, 3D, dull, flat"
}

last_used_seed = -1  # Global variable to store the last used seed for reproducibility

def upscale_image(image, scale_factor=2):
    """Upscale the image using OpenCV's Lanczos interpolation."""
    image_np = np.array(image)  # Convert PIL image to NumPy array
    height, width = image_np.shape[:2]  # Get original dimensions
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)  # Calculate new dimensions
    # Resize the image using Lanczos interpolation
    return Image.fromarray(cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4))

def decode_latents_to_image(latents):
    """
    Decode the latent tensor into a PIL image.
    This follows the standard post-processing used in Stable Diffusion.
    """
    latents = 1 / 0.18215 * latents  # Scale the latent tensor as per standard practice
    with torch.no_grad():
        image_tensor = pipe.vae.decode(latents).sample  # Decode the latent tensor using the VAE
    # Normalize and convert the tensor to an image
    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    image_np = image_tensor.cpu().permute(0, 2, 3, 1).numpy()
    image_np = (image_np[0] * 255).round().astype("uint8")
    return Image.fromarray(image_np)

def generate_image_non_streaming(prompt, negative_prompt, style, num_inference_steps, guidance_scale, image_width, image_height, seed):
    """
    Non-streaming image generation function.
    Generates the final image and returns it after completion.
    """
    # Start with the default negative prompt
    full_negative_prompt = DEFAULT_NEGATIVE_PROMPT

    # Append user's negative prompt if provided
    if negative_prompt:
        full_negative_prompt += f", {negative_prompt}"

    # Append style-specific negative prompt if a style is selected
    if style != "none":
        full_negative_prompt += f", {STYLE_NEGATIVE_PROMPTS[style]}"

    # Prepare the styled prompt by combining the user's prompt with the style prompt
    styled_prompt = f"{prompt}, {STYLE_PROMPTS[style]}"

    try:
        # Set the seed for reproducibility
        global last_used_seed  # Allow modification of the global variable
        if seed == -1:
            last_used_seed = torch.randint(0, 2**32 - 1, (1,)).item()  # Generate a random seed
        else:
            last_used_seed = seed  # Store the user-provided seed
        torch.manual_seed(last_used_seed)

        # Run the pipeline (without callback)
        result = pipe(
            prompt=styled_prompt,
            num_inference_steps=num_inference_steps,
            negative_prompt=full_negative_prompt,
            guidance_scale=guidance_scale,
            width=image_width,
            height=image_height,
        )

        return upscale_image(result.images[0], scale_factor=2)  # Return the upscaled final image

    except Exception as e:
        print(f"Generation error: {e}")
        return Image.new("RGB", (image_width, image_height), "red")  # Return a red error image

def generate_image(prompt, negative_prompt, style, num_inference_steps, guidance_scale, image_width, image_height, seed):
    """
    Generator function that streams intermediate images.
    A callback updates the global intermediate image.
    """
    # Start with the default negative prompt
    full_negative_prompt = DEFAULT_NEGATIVE_PROMPT

    # Append user's negative prompt if provided
    if negative_prompt:
        full_negative_prompt += f", {negative_prompt}"

    # Append style-specific negative prompt if a style is selected
    if style != "none":
        full_negative_prompt += f", {STYLE_NEGATIVE_PROMPTS[style]}"

    # Shared state variables to track the final image and completion status
    shared_state = {
        "final_image": None,
        "done": False
    }
    image_queue = queue.Queue()  # Queue to hold intermediate images

    def callback_fn(pipe, step, timestep, callback_kwargs):
        """Callback function to decode latents and update intermediate images."""
        if step % 10 == 0:  # Only decode every 10 steps to reduce overhead
            try:
                latents = callback_kwargs["latents"]
                current = decode_latents_to_image(latents)
                image_queue.put(current)  # Add the image to the queue
            except Exception as e:
                print(f"Callback decoding error at step {step}: {e}")
        return callback_kwargs
    
    def run_generation():
        """Run the image generation in a separate thread."""
        # Prepare the styled prompt
        styled_prompt = f"{prompt}, {STYLE_PROMPTS[style]}"
        try:
            # Set the seed for reproducibility
            global last_used_seed  # Allow modification of the global variable
            if seed == -1:
                last_used_seed = torch.randint(0, 2**32 - 1, (1,)).item()  # Generate a random seed
            else:
                last_used_seed = seed  # Store the user-provided seed
            torch.manual_seed(last_used_seed)
            
            # Run the pipeline with the callback
            result = pipe(
                prompt=styled_prompt,
                num_inference_steps=num_inference_steps,
                negative_prompt=full_negative_prompt,  # Use the full negative prompt
                guidance_scale=guidance_scale,
                width=image_width,
                height=image_height,
                callback_on_step_end=callback_fn,
                callback_on_step_end_timesteps=[i for i in range(num_inference_steps)],
            )
            shared_state["final_image"] = result.images[0]
        except Exception as e:
            print(f"Generation error: {e}")
            # In case of error, create a red error image
            shared_state["final_image"] = Image.new("RGB", (image_width, image_height), "red")
        shared_state["done"] = True
        image_queue.put(None)  # Signal that generation is done

    # Start the generation in a separate thread
    thread = threading.Thread(target=run_generation)
    thread.start()

    # Stream intermediate results until generation is done
    while True:
        try:
            current_image = image_queue.get(timeout=0.5)  # Wait for an image from the queue
            if current_image is None:  # Generation is done
                break
            yield upscale_image(current_image, scale_factor=2)  # Upscale and yield the intermediate image
        except queue.Empty:
            continue

    # Finally, yield the final image (upscaled)
    yield upscale_image(shared_state["final_image"], scale_factor=2)

# Define demo examples for the Gradio interface
demo_examples = [
    ["A majestic lion in the savannah", "", "realistic", 50, 8.0, 512, 512, -1],
    ["A futuristic cityscape at night", "blurry, low quality", "cinematic", 60, 10.0, 768, 512, -1],
    ["A cute cat wearing a hat", "", "realistic", 40, 7.0, 512, 512, -1],
    ["A fantasy dragon flying over mountains", "", "fantasy", 70, 12.0, 1024, 768, -1],
    ["A beautiful girl with pink hair", "", "cinematic", 50, 9.0, 512, 768, -1],
]

# Function to update width & height based on aspect ratio selection
def update_dimensions(aspect_ratio):
    aspect_ratios = {
        "1:1": (512, 512),
        "4:3": (768, 576),
        "3:2": (768, 512),
        "16:9": (1024, 576),
        "2:3": (512, 768),
        "9:16": (576, 1024),
    }
    
    # If the user selects a predefined ratio, return width, height & hide sliders
    if aspect_ratio in aspect_ratios:
        return aspect_ratios[aspect_ratio] + (gr.update(visible=False), gr.update(visible=False))
    
    # If "Custom" is selected, keep sliders visible and return empty updates
    return gr.update(), gr.update(), gr.update(visible=True), gr.update(visible=True)

# Function to fetch the current image's seed
def fetch_current_seed():
    global last_used_seed
    return last_used_seed  # Return the last used seed

# Function to set seed to -1 (random)
def set_random_seed():
    return -1  # Return -1 to indicate a random seed

# Build the Gradio interface with streaming output
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 Stable Diffusion XL Image Generator")
    with gr.Tabs():
        with gr.TabItem("Generate Image"):
            with gr.Row():
                with gr.Column():
                    prompt_input = gr.Textbox(label="Prompt", placeholder="Enter your prompt here", lines=3)
                    negative_prompt_input = gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompts here (optional)", lines=3)
                    style_dropdown = gr.Dropdown(label="Style", choices=STYLES, value="none", info="Choose a style for your image.")
                    steps_slider = gr.Slider(minimum=20, maximum=100, value=28, step=1, label="Inference Steps", info="More steps can improve quality but take longer.")
                    guidance_slider = gr.Slider(minimum=1.0, maximum=20.0, value=7.0, step=0.5, label="Guidance Scale", info="Higher values make the image more aligned with the prompt.")
                    
                    # Aspect ratio dropdown
                    aspect_ratio_dropdown = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=["Custom", "1:1", "4:3", "3:2", "16:9", "2:3", "9:16"],
                        value="Custom",
                        interactive=True
                    )

                    # Width & Height sliders (editable)
                    width_slider = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Image Width")
                    height_slider = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Image Height")

                   
                    seed_input = gr.Number(label="Seed", value=-1, interactive=True) 
                    fetch_seed_btn = gr.Button("🔄", elem_id="fetch-seed-btn", size="sm")
                    random_seed_btn = gr.Button("🎲", elem_id="random-seed-btn", size="sm")

                    # Add a toggle for streaming mode
                    stream_toggle = gr.Checkbox(label="Show Image Generation Process (Streaming)", value=False, info="Turn this on to watch the image being created step by step in real time. Turn it off to generate the final image faster without previewing the process.")

                    generate_button = gr.Button("Generate Image", variant="primary")

                    

                with gr.Column():
                    # The Image component will be updated with streaming outputs
                    image_output = gr.Image(label="Generated Image", streaming=True)
                    progress_bar = gr.Progress()

            with gr.Row():
                # Add demo examples
                gr.Examples(
                    examples=demo_examples,
                    inputs=[prompt_input, negative_prompt_input, style_dropdown, steps_slider, guidance_slider, width_slider, height_slider, seed_input],
                    label="Try these examples!"
                )
                    
    # Update the click event to use the selected mode
    def generate_wrapper(prompt, negative_prompt, style, num_inference_steps, guidance_scale, image_width, image_height, seed, stream_mode):
        if stream_mode:
            # If streaming is enabled, iterate over the generator and yield images one by one
            for img in generate_image(prompt, negative_prompt, style, num_inference_steps, guidance_scale, image_width, image_height, seed):
                yield img
        else:
            # If streaming is off, just return the final image
            yield generate_image_non_streaming(prompt, negative_prompt, style, num_inference_steps, guidance_scale, image_width, image_height, seed)

    generate_button.click(
        generate_wrapper,
        inputs=[prompt_input, negative_prompt_input, style_dropdown, steps_slider, guidance_slider, width_slider, height_slider, seed_input, stream_toggle],
        outputs=image_output,
    )

    # When a ratio is selected, update width & height and show/hide sliders
    aspect_ratio_dropdown.change(
        update_dimensions, 
        inputs=[aspect_ratio_dropdown], 
        outputs=[width_slider, height_slider, width_slider, height_slider]
    )

    fetch_seed_btn.click(fetch_current_seed, outputs=seed_input)
    random_seed_btn.click(lambda: -1, outputs=seed_input)  # Directly set -1 for randomness


demo.launch()