from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images, images
from modules.shared import opts, cmd_opts, state
from PIL import Image, ImageOps
from copy import deepcopy
from math import ceil
import cv2

import gradio as gr
import modules.scripts as scripts
from modules import sd_samplers
from random import randint
from skimage.util import random_noise
import numpy as np
import simple_depthmap as sdmg

class Script(scripts.Script):
    def title(self):
        return "Waifu harem generator"

    def ui(self, is_img2img):
        if is_img2img: return
        img2img_samplers_names = [s.name for s in sd_samplers.samplers_for_img2img]
        # foreground UI
        with gradio.Box():
            foregen_prompt      = gr.Textbox(label="Foreground prompt", lines=2, max_lines=2000)
            foregen_iter        = gr.Slider(minimum=1, maximum=10, step=1, label='Number of foreground images', value=1)
            foregen_steps       = gr.Slider(minimum=1, maximum=120, step=1, label='foreground steps', value=30)
            foregen_cfg_scale   = gr.Slider(minimum=1, maximum=30, step=0.1, label='foreground cfg scale', value=12)
            foregen_seed_shift  = gr.Slider(minimum=0, maximum=1000, step=1, label='foreground new seed+', value=1000)
            foregen_sampler     = gr.Dropdown(label="foreground sampler", choices=img2img_samplers_names, value="DDIM")
            foregen_clip        = gr.Slider(minimum=0, maximum=12, step=1, label='change clip for foreground (0 = no interaction)', value=0)
            with gr.Row():
                foregen_size_x  = gr.Slider(minimum=64, maximum=2048, step=64, label='foreground width (64 = same as background)', value=64)
                foregen_size_y  = gr.Slider(minimum=64, maximum=2048, step=64, label='foreground height (64 = same as background)', value=64)

        # blend UI
        with gradio.Box():
            foregen_blend_prompt             = gr.Textbox(label="final blend prompt", lines=2, max_lines=2000)
            foregen_blend_steps              = gr.Slider(minimum=1, maximum=120, step=1, label='blend steps', value=30)
            foregen_blend_cfg_scale          = gr.Slider(minimum=1, maximum=30, step=0.1, label='blend cfg scale', value=7.5)
            foregen_blend_denoising_strength = gr.Slider(minimum=0.1, maximum=1, step=0.01, label='blend denoising strength', value=0.3)
            foregen_blend_sampler            = gr.Dropdown(label="blend sampler", choices=img2img_samplers_names, value="DDIM")
            with gr.Row():
                foregen_blend_size_x  = gr.Slider(minimum=64, maximum=2048, step=64, label='blend height (64 = same size as background)', value=64)
                foregen_blend_size_y  = gr.Slider(minimum=64, maximum=2048, step=64, label='blend width  (64 = same size as background)', value=64)

        # Misc options
        foregen_x_shift  = gr.Slider(minimum=0, maximum=2, step=0.1, label='Foreground distance from center multiplier', value=1)
        foregen_y_shift  = gr.Slider(minimum=0, maximum=100, step=1, label='Foreground Y shift (far from center = lower)', value=0)
        foregen_treshold = gr.Slider(minimum=0, maximum=255, step=1, label='Foreground depth cut treshold', value=100)

        with gr.Row():
            foregen_save_background = gr.Checkbox(label='Save background', value=True)
            foregen_save_all        = gr.Checkbox(label='Save all foreground images', value=True)
            foregen_face_correction = gr.Checkbox(label='Face correction', value=True)
        return    []




    def run(self,p,):
        try:
            sdmg = sdmg() #import midas

            def cut_depth_mask(img,mask_img,foregen_treshold):
                mask_img = cv2.imread(mask_img, cv2.IMREAD_GRAYSCALE)
                mask_img = Image.fromarray(mask_img)
                mask_img = mask_img.convert("RGBA")
                mask_datas = mask_img.getdata()
                datas = img.getdata()

                treshold = foregen_treshold
                newData = []
                for i in range(len(mask_datas)):
                    if mask_datas[i][0] >= foregen_treshold and mask_datas[i][1] >= foregen_treshold and mask_datas[i][2] >= foregen_treshold:
                        newData.append(datas[i])
                    else:
                        newData.append((255, 255, 255, 0))
                mask_img.putdata(newData)
                return mask_img

            def paste_foreground(background,foreground,index,total_foreground,x_shift,y_shift):
                index = total_foreground-index-1
                image = Image.new("RGBA", background.size)
                image.paste(background, (0,0), background)
                alternator = -1 if index % 2 == 0 else 1
                if total_foreground % 2 == 0:
                    foreground_shift  = background.size[0]/2-foreground.size[0]/2 + background.size[0]/(total_foreground)*alternator*ceil(index/2)*x_shift - background.size[0]/(total_foreground)/2
                else:
                    index_shift = index-(index % 2)
                    if index == 0:
                        foreground_shift  = background.size[0]/2-foreground.size[0]/2
                    else:
                        foreground_shift  = background.size[0]/2-foreground.size[0]/2 + background.size[0]/(total_foreground)*alternator*ceil(index/2)*x_shift
                x_shift = int(foreground_shift)
                y_shift = ceil(index/2)*y_shift
                image.paste(foreground, (x_shift,background.size[1]-foreground.size[1]+y_shift), foreground)
                return image

            img2img_samplers_names = [s.name for s in sd_samplers.samplers_for_img2img]
            img2img_sampler_index = [i for i in range(len(img2img_samplers_names)) if img2img_samplers_names[i] == foregen_sampler][0]
            foregen_blend_sampler_index = [i for i in range(len(img2img_samplers_names)) if img2img_samplers_names[i] == foregen_blend_sampler][0]

            if p.seed == -1: p.seed = randint(0,1000000000)

            initial_CLIP = opts.data["CLIP_stop_at_last_layers"]
            orig_p = deepcopy(p)
            p.do_not_save_samples = not foregen_save_background

            n_iter=p.n_iter
            for j in range(n_iter):
                p.n_iter=1
                if foregen_clip > 0:
                    opts.data["CLIP_stop_at_last_layers"] = initial_CLIP
                proc = process_images(p) #background image processing
                background_image = proc.images[0]
                foregrounds = []
                if foregen_clip > 0:
                    opts.data["CLIP_stop_at_last_layers"] = foregen_clip
                for i in range(foregen_iter):
                    if state.interrupted:
                        if foregen_clip > 0:
                            opts.data["CLIP_stop_at_last_layers"] = initial_CLIP
                        break
                    p_bis = deepcopy(p)
                    p_bis.prompt    = foregen_prompt
                    p_bis.seed      = p.seed+foregen_seed_shift*(i+1)
                    p_bis.cfg_scale = foregen_cfg_scale
                    p_bis.steps     = foregen_steps
                    p_bis.do_not_save_samples = not foregen_save_all
                    p_bis.width     = foregen_size_x
                    p_bis.height    = foregen_size_y
                    foregen_proc    = process_images(p_bis)
                    foregrounds.append(foregen_proc.images[0])

                if foregen_clip > 0:
                    opts.data["CLIP_stop_at_last_layers"] = initial_CLIP

                for f in range(foregen_iter):
                    foreground_image      = foregrounds[f]
                    # gen depth map
                    foreground_image_mask = sdmg.calculate_depth_map_for_waifus(foreground_image)
                    # cut depth
                    foreground_image      = cut_depth_mask(foreground_image,foreground_image_mask,foregen_treshold)
                    # paste foregrounds onto background
                    background_image      = paste_foreground(background_image,foreground_image,f,foregen_iter,foregen_x_shift,foregen_y_shift)

                # final blend
                img2img_processing = StableDiffusionProcessingImg2Img(
                    init_images=[background_image],
                    resize_mode=0,
                    denoising_strength=foregen_blend_denoising_strength,
                    mask=None,
                    mask_blur=4,
                    inpainting_fill=0,
                    inpaint_full_res=True,
                    inpaint_full_res_padding=0,
                    inpainting_mask_invert=0,
                    sd_model=p.sd_model,
                    outpath_samples=p.outpath_samples,
                    outpath_grids=p.outpath_grids,
                    prompt=foregen_blend_prompt,
                    styles=p.styles,
                    seed=p.seed+foregen_seed_shift*(foregen_iter+1),
                    subseed=proc_temp.subseed,
                    subseed_strength=p.subseed_strength,
                    seed_resize_from_h=p.seed_resize_from_h,
                    seed_resize_from_w=p.seed_resize_from_w,
                    sampler_index=foregen_blend_sampler_index,
                    batch_size=p.batch_size,
                    n_iter=p.n_iter,
                    steps=foregen_blend_steps,
                    cfg_scale=foregen_blend_cfg_scale,
                    width=foregen_blend_size_x,
                    height=foregen_blend_size_y,
                    restore_faces=foregen_face_correction,
                    tiling=p.tiling,
                    do_not_save_samples=orig_p.do_not_save_samples,
                    do_not_save_grid=p.do_not_save_grid,
                    extra_generation_params=p.extra_generation_params,
                    overlay_images=p.overlay_images,
                    negative_prompt=p.negative_prompt,
                    eta=p.eta
                    )
                final_blend = process_images(img2img_processing)

                p.seed+=1
            return final_blend
        except Exception as e:
            pass
        finally:
            opts.data["CLIP_stop_at_last_layers"] = initial_CLIP
            sdmg.del_model()