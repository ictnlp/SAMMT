import torch
import torch.nn.functional as F
import os
from os.path import join
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import numpy as np
from diffusers import LMSDiscreteScheduler

torch_device = "cuda:3" if torch.cuda.is_available() else "cpu"
split = 'train'
dic = {
    'train': 'flickr30k-images',
    'valid': 'flickr30k-images',
    'test': 'flickr30k-images',
    'test1': 'test_2017_flickr',
    'test2': 'test_2017_mscoco'
}
dic1 = {
    'train': 'train',
    'valid': 'val',
    'test': 'test_2016_flickr',
    'test1': 'test_2017_flickr',
    'test2': 'test_2017_mscoco'
}

imagepth = join('flickr30k', dic[split])
sdimagepth = join('flickr30k-sdimages', dic[split])
if not os.path.exists(sdimagepth):
    os.makedirs(sdimagepth)

imagenamepth = join('multi30k-dataset/data/task1/image_splits',dic1[split] + '.txt')  
textpth = join('multi30k-dataset/data/task1/tok', dic1[split] + '.lc.norm.tok.en')

# 1. Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained(
    "/root/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/59ec6bdf37d6279d3c0faf36e89ff1aa34f7ebf4",
    subfolder="vae")  # CompVis/stable-diffusion-v1-4

# 2. Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained(
    "/root/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff") # openai/clip-vit-large-patch14
text_encoder = CLIPTextModel.from_pretrained(
    "/root/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff")  # openai/clip-vit-large-patch14

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained(
    "/root/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/59ec6bdf37d6279d3c0faf36e89ff1aa34f7ebf4",
    subfolder="unet") # "CompVis/stable-diffusion-v1-4"

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",num_train_timesteps=1000)

vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)

height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion

num_inference_steps = 50  # Number of denoising steps

guidance_scale = 7.5  # Scale for classifier-free guidance

generator = torch.manual_seed(0)  # Seed generator to create the inital latent noise

batch_size = 1


def main():
    name_inputs=[]
    with open(imagenamepth, 'r', encoding='utf-8') as src_file:
        for line in src_file.readlines():
            line=line.split("#")
            name_inputs.append(line[0].strip())  # name_inputs = list(map(str.strip, src_file.readlines()))
    with open(textpth, 'r', encoding='utf-8') as src_file:
        text_inputs = list(map(str.strip, src_file.readlines()))
    chunk_size = 1
    for chunk_id in range(2608, 7000):  # ,len(name_inputs) // chunk_size + 1
        begin = chunk_id * chunk_size
        end = min((chunk_id + 1) * chunk_size, len(name_inputs))
        for idx in range(begin, end):
            print('{0}/{1}'.format(idx, len(name_inputs)))
            prompt = str(text_inputs[idx])
            text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                                   truncation=True, return_tensors="pt")
            with torch.no_grad():
                text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
            max_length = text_input.input_ids.shape[-1]
            uncond_input = tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            with torch.no_grad():
                uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            latents = torch.randn(
                (batch_size, unet.in_channels, height // 8, width // 8),
                generator=generator,
            )
            latents = latents.to(torch_device)
            scheduler.set_timesteps(num_inference_steps)
            latents = latents * scheduler.init_noise_sigma
            from tqdm.auto import tqdm
            from torch import autocast
            for t in tqdm(scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                # predict the noise residual
                with torch.no_grad():
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            # scale and decode the image latents with vae
            latents = (1 / 0.18215 * latents).to(torch_device)
            with torch.no_grad():
                image = vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            fname = join(sdimagepth, name_inputs[idx])
            pil_images[0].save(fname)

if __name__ == '__main__':
    main()