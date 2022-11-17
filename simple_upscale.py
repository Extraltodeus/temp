
import math
import os
import sys
import traceback
import random

import modules.scripts as scripts
import modules.images as images
import gradio as gr

from modules.processing import Processed, process_images
from PIL import Image
from modules.shared import opts, cmd_opts, state

class Script(scripts.Script):
    alwayson = True
    def title(self):
        return "Lanczos simple upscale"
      
    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        upscale_factor = gr.Slider(minimum=1, maximum=4, step=0.1, label='Upscale factor', value=2)
        return [upscale_factor]
      
    def batch_postprocess(self, p, image, *args, **kwargs):
        def simple_upscale(img, upscale_factor):
            w, h = img.size
            w = int(w * upscale_factor)
            h = int(h * upscale_factor)
            return img.resize((w, h), Image.Resampling.LANCZOS)
        simple_upscale(image, upscale_factor)
        images.save_image(image, p.outpath_samples, "", p=p)