
import math
import os
import sys
import traceback
import random
from copy import deepcopy

import modules.scripts as scripts
from   modules.script_callbacks import on_before_image_saved
import modules.face_restoration
import modules.images as images
import gradio as gr

from modules.processing import Processed, process_images
from PIL import Image
from modules.shared import opts, cmd_opts, state
import numpy as np


class Script(scripts.Script):
    alwayson = True

    def __init__(self):
        on_before_image_saved(self.bis)

    def process(self, p, simple_upscale_factor,multi_face_correction):
        p.simple_upscale_factor = simple_upscale_factor
        p.multi_face_correction = multi_face_correction
        p.initial_image_check   = ""

    def title(self):
        return "Lanczos simple upscale"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Row():
            simple_upscale_factor = gr.Slider(minimum=1, maximum=2, step=0.1, label='Upscale factor ', value=1)
            multi_face_correction = gr.Slider(minimum=0, maximum=10, step=1, label='Extra face restorations', value=0)
        return [simple_upscale_factor,multi_face_correction]

    def bis(self, params):
        try:
            if params.image != params.p.initial_image_check:
                if params.p.multi_face_correction > 0:
                    x_sample = np.asarray(params.image)
                    for c in range(params.p.multi_face_correction):
                        x_sample = modules.face_restoration.restore_faces(x_sample)
                        print("restoring face :",c+1,"/",params.p.multi_face_correction)
                    params.image = Image.fromarray(x_sample)
                if params.p.simple_upscale_factor > 1:
                    w, h = params.image.size
                    w = int(w * params.p.simple_upscale_factor)
                    h = int(h * params.p.simple_upscale_factor)
                    image = params.image.resize((w, h), Image.Resampling.LANCZOS)
                    params.image = image
                params.p.initial_image_check = deepcopy(params.image)
        except Exception:
            pass
