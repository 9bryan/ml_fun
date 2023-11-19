import torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline
from pywebio import start_server
from pywebio.input import input, FLOAT, TEXT
from pywebio.output import put_text, put_image, scroll_to

#pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=False)
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.safety_checker = None
captioner = pipeline("image-to-text",model="ydshieh/vit-gpt2-coco-en")

def ml_telephone():
    prompt = input("Enter your initial prompt", type=TEXT)
    put_text('Your initial prompt: %s' % (prompt))

    while True:
        image=pipe(prompt).images[0]
        put_image(image)
        caption=captioner(image)[0]["generated_text"]
        put_text(caption)
        prompt=caption
        scroll_to(position='bottom')

if __name__ == '__main__':
    start_server(ml_telephone, port=8080)
