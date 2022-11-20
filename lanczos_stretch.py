
import math
import os
import sys
import traceback
import random

import modules.scripts as scripts
from   modules.script_callbacks import on_before_image_saved
import modules.images as images
import gradio as gr

from modules.processing import Processed, process_images
from PIL import Image
from modules.shared import opts, cmd_opts, state

class Script(scripts.Script):
    alwayson = True
    
    def __init__(self):
        on_before_image_saved(self.bis)
    
    def title(self):
        return "Lanczos simple upscale"
      
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        simple_upscale_factor = gr.Slider(minimum=1, maximum=4, step=0.1, label='Upscale factor', value=2)
        return [simple_upscale_factor]
          
    def bis(self, params):
        if ImageSaveParams.p.simple_upscale_factor > 1:
            w, h = ImageSaveParams.image.size
            w = int(w * ImageSaveParams.p.simple_upscale_factor)
            h = int(h * ImageSaveParams.p.simple_upscale_factor)
            image = ImageSaveParams.image.resize((w, h), Image.Resampling.LANCZOS)
            params.image = image
#             images.save_image(image, ImageSaveParams.p.outpath_samples, ImageSaveParams.filename, ImageSaveParams.p.seed, ImageSaveParams.p.prompt, opts.samples_format, info=ImageSaveParams.pnginfo, p=ImageSaveParams.p)
