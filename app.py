import torch
import streamlit as st
from diffusers import DiffusionPipeline
from PIL import Image
import io

# Title of the app
st.title("Text to Image Generator using Stable Diffusion")

# Prompt input
prompt = st.text_input("Enter a prompt for image generation:", "a serene landscape with mountains and a river at sunset")

# Load the stable diffusion pipeline once (only when the app starts)
@st.cache_resource
def load_model():
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
    pipe = pipe.to("cuda")  # Move the model to GPU if available
    return pipe

# Button to trigger image generation
if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        # Load the model
        pipe = load_model()
        
        # Generate image
        image = pipe(prompt).images[0]
        
        # Display the image
        st.image(image, caption="Generated Image", use_column_width=True)

        # Save the generated image as a downloadable file
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Download Image",
            data=byte_im,
            file_name="generated_image.png",
            mime="image/png"
        )
