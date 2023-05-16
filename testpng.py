from modules.shared import opts, cmd_opts, state
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images, images
from PIL import Image, ImageDraw, PngImagePlugin
import gradio as gr
import modules.scripts as scripts
from modules import sd_samplers
from random import randint
from skimage.util import random_noise
import base64
import io
import random
import numpy as np
import os

def remap_range(value, minIn, MaxIn, minOut, maxOut):
            if value > MaxIn: value = MaxIn;
            if value < minIn: value = minIn;
            if (MaxIn - minIn) == 0 : return maxOut
            finalValue = ((value - minIn) / (MaxIn - minIn)) * (maxOut - minOut) + minOut;
            return finalValue;

class ImageEmbedder:
    @staticmethod
    def encode_image_base64(image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        encoded_image = base64.b64encode(buffered.getvalue())
        return encoded_image.decode("utf-8")

    @staticmethod
    def embed_image_in_metadata(encoded_original_image, replacement_image, output_image_path):
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("embedded_image", encoded_original_image)
        replacement_image.save(output_image_path, "PNG", pnginfo=metadata)

    @staticmethod
    def mix_string(s, seed):
        s_list = list(s)
        random.seed(seed, version=2)
        random.shuffle(s_list)
        mixed_string = ''.join(s_list)
        return mixed_string

class Script(scripts.Script):
    def __init__(self):
        self.rep_images_folder_path = """/kaggle/temp/AI_PICS/out1/sources_inspiration"""
        self.rep_images = os.listdir(self.rep_images_folder_path)

    def title(self):
        return "Txt2img2img2img_patch_upscale_uboot"

    def ui(self, is_img2img):
        if is_img2img: return
        img2img_samplers_names = [s.name for s in sd_samplers.samplers_for_img2img]
        t2iii_reprocess = gr.Slider(minimum=1, maximum=128, step=1, label='Number of img2img', value=1)
        t2iii_steps = gr.Slider(minimum=1, maximum=120, step=1, label='img2img steps', value=42)
        t2iii_cfg_scale = gr.Slider(minimum=1, maximum=30, step=0.1, label='img2img cfg scale ', value=8.3)
        t2iii_seed_shift = gr.Slider(minimum=-1, maximum=1000000, step=1, label='img2img new seed+ (-1 for random)', value=1)
        t2iii_denoising_strength = gr.Slider(minimum=0, maximum=1, step=0.01, label='img2img denoising strength ', value=0.42)
        t2iii_2x_last = gr.Slider(minimum=1, maximum=4, step=0.1, label='resize time x size for last pass', value=1)
        with gr.Row():
            t2iii_patch_upscale = gr.Checkbox(label='Patch upscale', value=False)
            t2iii_save_first    = gr.Checkbox(label='Save first image', value=False)
            t2iii_only_last     = gr.Checkbox(label='Only save last img2img', value=True)
            t2iii_face_correction = gr.Checkbox(label='Face correction on all', value=False)
            t2iii_face_correction_last = gr.Checkbox(label='Face correction on last', value=True)

        t2iii_sampler = gr.Dropdown(label="Sampler", choices=img2img_samplers_names, value="DDIM")
        t2iii_clip    = gr.Slider(minimum=0, maximum=12, step=1, label='change clip for img2img (0 = disabled)', value=0)
        t2iii_noise   = gr.Slider(minimum=0, maximum=0.05,  step=0.001, label='Add noise before img2img ', value=0)
        t2iii_patch_padding = gr.Slider(minimum=0, maximum=512,  step=2, label='Patch upscale padding', value=32)
        t2iii_patch_square_size = gr.Slider(minimum=64, maximum=1024,  step=64, label='Patch upscale square size', value=512)
        t2iii_patch_border      = gr.Slider(minimum=0, maximum=256,  step=1, label='Patch upscale mask border', value=32)
        t2iii_patch_mask_blur   = gr.Slider(minimum=0, maximum=64,  step=1, label='Patch upscale mask blur', value=4)
        t2iii_patch_end_denoising   = gr.Slider(minimum=0, maximum=1,  step=0.01, label='Patch end denoising', value=0)
        t2iii_upscale_x = gr.Slider(minimum=64, maximum=8192, step=64, label='img2img width (64 = no rescale)', value=64)
        t2iii_upscale_y = gr.Slider(minimum=64, maximum=8192, step=64, label='img2img height (64 = no rescale)', value=64)
        with gr.Row():
            uboot_mode   = gr.Checkbox(label='U-boot', value=False)
            uboot_enigma = gr.Textbox(label="Enigma", lines=1, max_lines=1)
        return    [t2iii_reprocess,t2iii_steps,t2iii_cfg_scale,t2iii_seed_shift,t2iii_denoising_strength,t2iii_patch_upscale,t2iii_2x_last,t2iii_save_first,t2iii_only_last,t2iii_face_correction,t2iii_face_correction_last,t2iii_sampler,t2iii_clip,t2iii_noise,t2iii_patch_padding,t2iii_patch_square_size,t2iii_patch_border,t2iii_patch_mask_blur,t2iii_patch_end_denoising,t2iii_upscale_x,t2iii_upscale_y,uboot_mode,uboot_enigma]
    def run(self,p,t2iii_reprocess,t2iii_steps,t2iii_cfg_scale,t2iii_seed_shift,t2iii_denoising_strength,t2iii_patch_upscale,t2iii_2x_last,t2iii_save_first,t2iii_only_last,t2iii_face_correction,t2iii_face_correction_last,t2iii_sampler,t2iii_clip,t2iii_noise,t2iii_patch_padding,t2iii_patch_square_size,t2iii_patch_border,t2iii_patch_mask_blur,t2iii_patch_end_denoising,t2iii_upscale_x,t2iii_upscale_y,uboot_mode,uboot_enigma):
        def add_noise_to_image(img,seed,t2iii_noise):
            img = np.array(img)
            img = random_noise(img, mode='gaussian', seed=proc.seed, clip=True, var=t2iii_noise)
            img = np.array(255*img, dtype = 'uint8')
            img = Image.fromarray(np.array(img))
            return img
        def create_mask(size, border_width):
            im = Image.new('RGB', (size, size), color='white')
            draw = ImageDraw.Draw(im)
            draw.rectangle((0, 0, size-1, size-1), outline='black', width=border_width)
            return im

        img2img_samplers_names = [s.name for s in sd_samplers.samplers_for_img2img]
        img2img_sampler_index = [i for i in range(len(img2img_samplers_names)) if img2img_samplers_names[i] == t2iii_sampler][0]
        if p.seed == -1: p.seed = randint(0,1000000000)

        initial_CLIP = opts.data["CLIP_stop_at_last_layers"]
        p.do_not_save_samples = not t2iii_save_first or uboot_mode

        n_iter=p.n_iter
        for j in range(n_iter):
            p.n_iter=1
            if t2iii_clip > 0:
                opts.data["CLIP_stop_at_last_layers"] = initial_CLIP
            proc = process_images(p)
            basename = ""
            extra_gen_parms = {
            'Initial steps':p.steps,
            'Initial CFG scale':p.cfg_scale,
            "Initial seed": p.seed,
            'Initial sampler': p.sampler_name,
            'Reprocess amount':t2iii_reprocess
            }
            for i in range(t2iii_reprocess):
                if t2iii_upscale_x > 64:
                    upscale_x = t2iii_upscale_x
                else:
                    upscale_x = p.width
                if t2iii_upscale_y > 64:
                    upscale_y = t2iii_upscale_y
                else:
                    upscale_y = p.height
                if t2iii_2x_last > 1 and i+1 == t2iii_reprocess:
                    upscale_x = int(upscale_x*t2iii_2x_last)
                    upscale_y = int(upscale_y*t2iii_2x_last)
                if t2iii_seed_shift == -1:
                    reprocess_seed = randint(0,999999999)
                else:
                    reprocess_seed = p.seed+t2iii_seed_shift*(i+1)
                if t2iii_clip > 0:
                    opts.data["CLIP_stop_at_last_layers"] = t2iii_clip
                if state.interrupted:
                    if t2iii_clip > 0:
                        opts.data["CLIP_stop_at_last_layers"] = initial_CLIP
                    break
                if i == 0:
                    proc_temp = proc
                else:
                    proc_temp = proc2
                if t2iii_noise > 0 :
                    proc_temp.images[0] = add_noise_to_image(proc_temp.images[0],p.seed,t2iii_noise)

                img2img_processing = StableDiffusionProcessingImg2Img(
                    init_images=proc_temp.images,
                    resize_mode=0,
                    denoising_strength=remap_range(i,0,t2iii_reprocess-1,t2iii_denoising_strength,t2iii_patch_end_denoising) if t2iii_patch_end_denoising > 0 else t2iii_denoising_strength,
                    mask=None,
                    mask_blur=t2iii_patch_mask_blur,
                    inpainting_fill=1,
                    inpaint_full_res=True,
                    inpaint_full_res_padding=0,
                    inpainting_mask_invert=0,
                    sd_model=p.sd_model,
                    outpath_samples=p.outpath_samples,
                    outpath_grids=p.outpath_grids,
                    prompt=proc.info.split("\nNegative prompt")[0],
                    styles=p.styles,
                    seed=reprocess_seed,
                    subseed=proc_temp.subseed,
                    subseed_strength=p.subseed_strength,
                    seed_resize_from_h=p.seed_resize_from_h,
                    seed_resize_from_w=p.seed_resize_from_w,
                    #seed_enable_extras=p.seed_enable_extras,
                    sampler_name=t2iii_sampler,
                    #sampler_index=img2img_sampler_index,
                    batch_size=p.batch_size,
                    n_iter=p.n_iter,
                    steps=t2iii_steps,
                    cfg_scale=t2iii_cfg_scale,
                    width=upscale_x,
                    height=upscale_y,
                    restore_faces=(t2iii_face_correction or (t2iii_face_correction_last and t2iii_reprocess-1 == i)) and not (t2iii_reprocess-1 == i and not t2iii_face_correction_last),
                    tiling=p.tiling,
                    do_not_save_samples=not ((t2iii_only_last and t2iii_reprocess-1 == i) or not t2iii_only_last) or uboot_mode,
                    do_not_save_grid=p.do_not_save_grid,
                    extra_generation_params=extra_gen_parms,
                    overlay_images=p.overlay_images,
                    negative_prompt=p.negative_prompt,
                    eta=p.eta
                    )
                if not t2iii_patch_upscale:
                    proc2 = process_images(img2img_processing)
                else:
                    proc_temp.images[0] = proc_temp.images[0].resize((upscale_x, upscale_y), Image.Resampling.LANCZOS)
                    width_for_patch, height_for_patch = proc_temp.images[0].size
                    real_square_size = int(t2iii_patch_square_size-2*t2iii_patch_padding)
                    overlap_pass = int(real_square_size/t2iii_reprocess)*i
                    patch_seed = reprocess_seed
                    for x in range(0, width_for_patch+overlap_pass, real_square_size):
                        for y in range(0, height_for_patch+overlap_pass, real_square_size):
                            if t2iii_seed_shift == -1:
                                patch_seed = randint(0,999999999)
                            else:
                                patch_seed = patch_seed+t2iii_seed_shift
                            patch = proc_temp.images[0].crop((x-t2iii_patch_padding-overlap_pass, y-t2iii_patch_padding-overlap_pass, x + real_square_size + t2iii_patch_padding-overlap_pass, y + real_square_size + t2iii_patch_padding-overlap_pass))
                            img2img_processing.init_images = [patch]
                            img2img_processing.do_not_save_samples = True
                            img2img_processing.width  = patch.size[0]
                            img2img_processing.height = patch.size[1]
                            img2img_processing.seed   = patch_seed
                            mask = create_mask(patch.size[0],t2iii_patch_border)
                            img2img_processing.image_mask = mask
                            proc_patch_temp = process_images(img2img_processing)
                            patch = proc_patch_temp.images[0]
                            patch = patch.crop((t2iii_patch_padding, t2iii_patch_padding, patch.size[0] - t2iii_patch_padding, patch.size[1] - t2iii_patch_padding))
                            proc_temp.images[0].paste(patch, (x-overlap_pass, y-overlap_pass))
                    proc2 = proc_patch_temp
                    proc2.images[0] = proc_temp.images[0]
                    images.save_image(proc2.images[0], p.outpath_samples, "", proc2.seed, proc2.prompt, opts.samples_format, info=proc2.info, p=p)
                if uboot_mode:
                    original_image = proc2.images[0]
                    encoded_original_image = ImageEmbedder.encode_image_base64(original_image)
                    encoded_original_image_mixed = ImageEmbedder.mix_string(encoded_original_image,uboot_enigma)
                    rep_image_path = os.path.join(self.rep_images_folder_path,random.choice(self.rep_images))
                    rep_image = Image.open(rep_image_path)
                    output_image_name = f"00000-{p.seed}-{p.sampler_name}-{p.steps}-{p.cfg_scale}-{state.job_timestamp}.png"
                    output_image_path = os.path.join(p.outpath_samples,output_image_name)
                    # print("p.outpath_samples",p.outpath_samples)
                    # print("opts.samples_format",opts.samples_format)
                    ImageEmbedder.embed_image_in_metadata(encoded_original_image_mixed, rep_image, output_image_path)

            p.subseed = p.subseed + 1 if p.subseed_strength  > 0 else p.subseed
            p.seed    = p.seed    + 1 if p.subseed_strength == 0 else p.seed
        if t2iii_clip > 0:
            opts.data["CLIP_stop_at_last_layers"] = initial_CLIP
        return proc
