
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
    def title(self):
        return "totoprinter"

    def ui(self, is_img2img):
        toto = gr.Textbox(label="Toto", lines=1, value="Toto")
        return [toto]

    def run(self, p, toto):
        print(toto)
        return
