from modules.shared import opts, cmd_opts, state
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images, images
from PIL import Image
import gradio as gr
import modules.scripts as scripts
from modules import sd_samplers
from random import randint
from skimage.util import random_noise
import numpy as np

class Script(scripts.Script):
    def title(self):
        return "Txt2img2img2img"

    def ui(self, is_img2img):
        if is_img2img: return
        img2img_samplers_names = [s.name for s in sd_samplers.samplers_for_img2img]
        t2iii_reprocess = gr.Slider(minimum=1, maximum=10, step=1, label='Number of img2img', value=3)
        t2iii_steps = gr.Slider(minimum=1, maximum=120, step=1, label='img2img steps', value=30)
        t2iii_cfg_scale = gr.Slider(minimum=1, maximum=30, step=0.1, label='img2img cfg scale', value=12)
        t2iii_seed_shift = gr.Slider(minimum=0, maximum=1000000, step=1, label='img2img new seed+', value=1000)
        t2iii_denoising_strength = gr.Slider(minimum=0, maximum=1, step=0.01, label='img2img denoising strength ', value=0.4)
        with gr.Row():
            t2iii_save_first = gr.Checkbox(label='Save first image', value=False)
            t2iii_only_last = gr.Checkbox(label='Only save last img2img', value=True)
            t2iii_face_correction = gr.Checkbox(label='Face correction on all', value=False)
            t2iii_face_correction_last = gr.Checkbox(label='Face correction on last', value=True)

        t2iii_sampler = gr.Dropdown(label="Sampler", choices=img2img_samplers_names, value="DDIM")
        t2iii_clip    = gr.Slider(minimum=0, maximum=12, step=1, label='change clip for img2img (0 = disabled)', value=0)
        t2iii_noise   = gr.Slider(minimum=0, maximum=0.05,  step=0.001, label='Add noise before img2img ', value=0)
        t2iii_upscale_x = gr.Slider(minimum=64, maximum=2048, step=64, label='img2img width (64 = no rescale)', value=64)
        t2iii_upscale_y = gr.Slider(minimum=64, maximum=2048, step=64, label='img2img height (64 = no rescale)', value=64)
        return    [t2iii_reprocess,t2iii_steps,t2iii_cfg_scale,t2iii_seed_shift,t2iii_denoising_strength,t2iii_save_first,t2iii_only_last,t2iii_face_correction,t2iii_face_correction_last,t2iii_sampler,t2iii_clip,t2iii_noise,t2iii_upscale_x,t2iii_upscale_y]

    def run(self,p,t2iii_reprocess,t2iii_steps,t2iii_cfg_scale,t2iii_seed_shift,t2iii_denoising_strength,t2iii_save_first,t2iii_only_last,t2iii_face_correction,t2iii_face_correction_last,t2iii_sampler,t2iii_clip,t2iii_noise,t2iii_upscale_x,t2iii_upscale_y):

        def add_noise_to_image(img,seed,t2iii_noise):
            img = np.array(img)
            img = random_noise(img, mode='gaussian', seed=proc.seed, clip=True, var=t2iii_noise)
            img = np.array(255*img, dtype = 'uint8')
            img = Image.fromarray(np.array(img))
            return img

        img2img_samplers_names = [s.name for s in sd_samplers.samplers_for_img2img]
        img2img_sampler_index = [i for i in range(len(img2img_samplers_names)) if img2img_samplers_names[i] == t2iii_sampler][0]
        if p.seed == -1: p.seed = randint(0,1000000000)

        initial_CLIP = opts.data["CLIP_stop_at_last_layers"]
        p.do_not_save_samples = not t2iii_save_first

        if t2iii_upscale_x > 64:
            upscale_x = t2iii_upscale_x
        else:
            upscale_x = p.width
        if t2iii_upscale_y > 64:
            upscale_y = t2iii_upscale_y
        else:
            upscale_y = p.height

        n_iter=p.n_iter
        for j in range(n_iter):
            p.n_iter=1
            if t2iii_clip > 0:
                opts.data["CLIP_stop_at_last_layers"] = initial_CLIP
            proc = process_images(p)
            basename = ""
            for i in range(t2iii_reprocess):
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
                    denoising_strength=t2iii_denoising_strength,
                    mask=None,
                    mask_blur=4,
                    inpainting_fill=0,
                    inpaint_full_res=True,
                    inpaint_full_res_padding=0,
                    inpainting_mask_invert=0,
                    sd_model=p.sd_model,
                    outpath_samples=p.outpath_samples,
                    outpath_grids=p.outpath_grids,
                    prompt=proc.info.split("\nNegative prompt")[0],
                    styles=p.styles,
                    seed=p.seed+t2iii_seed_shift*(i+1),
                    subseed=proc_temp.subseed,
                    subseed_strength=p.subseed_strength,
                    seed_resize_from_h=p.seed_resize_from_h,
                    seed_resize_from_w=p.seed_resize_from_w,
                    #seed_enable_extras=p.seed_enable_extras,
                    sampler_name=t2iii_sampler,
#                     sampler_index=img2img_sampler_index,
                    batch_size=p.batch_size,
                    n_iter=p.n_iter,
                    steps=t2iii_steps,
                    cfg_scale=t2iii_cfg_scale,
                    width=upscale_x,
                    height=upscale_y,
                    restore_faces=t2iii_face_correction or (t2iii_face_correction_last and t2iii_reprocess-1 == i),
                    tiling=p.tiling,
                    do_not_save_samples=not ((t2iii_only_last and t2iii_reprocess-1 == i) or not t2iii_only_last),
                    do_not_save_grid=p.do_not_save_grid,
                    extra_generation_params=p.extra_generation_params,
                    overlay_images=p.overlay_images,
                    negative_prompt=p.negative_prompt,
                    eta=p.eta
                    )
                proc2 = process_images(img2img_processing)
            p.seed+=1
        if t2iii_clip > 0:
            opts.data["CLIP_stop_at_last_layers"] = initial_CLIP
        return proc
