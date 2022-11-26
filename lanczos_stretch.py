
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

    def process(self, p, simple_upscale_factor):
        p.simple_upscale_factor = simple_upscale_factor

    def title(self):
        return "Lanczos simple upscale"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        simple_upscale_factor = gr.Slider(minimum=1, maximum=4, step=0.1, label='Upscale factor', value=1)
        return [simple_upscale_factor]

    def bis(self, params):
        try:
            if params.p.simple_upscale_factor > 1:
                w, h = params.image.size
                w = int(w * params.p.simple_upscale_factor)
                h = int(h * params.p.simple_upscale_factor)
                image = params.image.resize((w, h), Image.Resampling.LANCZOS)
                params.image = image
        except Exception:
            pass
