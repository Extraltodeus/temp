from modules.shared import opts, cmd_opts, state
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images, images
from PIL import Image
import gradio as gr
import modules.scripts as scripts
from modules import sd_samplers
from random import randint

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
        t2iii_denoising_strength = gr.Slider(minimum=0.1, maximum=1, step=0.01, label='img2img denoising strength', value=0.4)
        t2iii_upscale_factor = gr.Slider(minimum=1, maximum=4, step=0.1, label='Stretch before save (factor, 1=disabled)', value=2)
        t2iii_only_last = gr.Checkbox(label='Only save the last img2img', value=True)
        t2iii_sampler = gr.Dropdown(label="Sampler", choices=img2img_samplers_names, value="DDIM")
        t2iii_upscale_x = gr.Slider(minimum=64, maximum=2048, step=64, label='img2img width (64 = no rescale)', value=64)
        t2iii_upscale_y = gr.Slider(minimum=64, maximum=2048, step=64, label='img2img height (64 = no rescale)', value=64)
        return [t2iii_reprocess,t2iii_steps,t2iii_cfg_scale,t2iii_seed_shift,t2iii_denoising_strength,t2iii_upscale_factor,t2iii_only_last,t2iii_sampler,t2iii_upscale_x,t2iii_upscale_y]

    def run(self,p,t2iii_reprocess,t2iii_steps,t2iii_cfg_scale,t2iii_seed_shift,t2iii_denoising_strength,t2iii_upscale_factor,t2iii_only_last,t2iii_sampler,t2iii_upscale_x,t2iii_upscale_y):
        img2img_samplers_names = [s.name for s in sd_samplers.samplers_for_img2img]
        img2img_sampler_index = [i for i in range(len(img2img_samplers_names)) if img2img_samplers_names[i] == t2iii_sampler][0]
        if p.seed == -1: p.seed = randint(0,1000000000)

        def simple_upscale(img, factor):
            w, h = img.size
            w = int(w * factor)
            h = int(h * factor)
            return img.resize((w, h), Image.LANCZOS)

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
            proc = process_images(p)
            basename = ""
            for i in range(t2iii_reprocess):
                if state.interrupted:
                    break
                if i == 0:
                    proc_temp = proc
                else:
                    proc_temp = proc2
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
                    seed=p.seed+t2iii_seed_shift,
                    subseed=proc_temp.subseed,
                    subseed_strength=p.subseed_strength,
                    seed_resize_from_h=p.seed_resize_from_h,
                    seed_resize_from_w=p.seed_resize_from_w,
                    #seed_enable_extras=p.seed_enable_extras,
                    sampler_index=img2img_sampler_index,
                    batch_size=p.batch_size,
                    n_iter=p.n_iter,
                    steps=t2iii_steps,
                    cfg_scale=t2iii_cfg_scale,
                    width=upscale_x,
                    height=upscale_y,
                    restore_faces=p.restore_faces,
                    tiling=p.tiling,
                    do_not_save_samples=True,
                    do_not_save_grid=p.do_not_save_grid,
                    extra_generation_params=p.extra_generation_params,
                    overlay_images=p.overlay_images,
                    negative_prompt=p.negative_prompt,
                    eta=p.eta
                    )
                proc2 = process_images(img2img_processing)
                if (t2iii_only_last and t2iii_reprocess-1 == i) or not t2iii_only_last :
                    if t2iii_upscale_factor > 1:
                        image = simple_upscale(proc2.images[0],t2iii_upscale_factor)
                    else:
                        image = proc2.images[0]
                    images.save_image(image, p.outpath_samples, "", proc2.seed, proc2.prompt, opts.samples_format, info= proc2.info, p=p)
            p.seed+=1
        return proc
