from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images, images, fix_seed
from modules.shared import opts, cmd_opts, state
from PIL import Image, ImageOps
from math import ceil
import cv2

import modules.scripts as scripts
from modules import sd_samplers
from random import randint, shuffle
import random
from skimage.util import random_noise
import gradio as gr
import numpy as np
import sys
import os
import importlib.util

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Script(scripts.Script):
    def title(self):
        return "Depth aware img2img mask"

    def ui(self, is_img2img):
        if not is_img2img: return
        save_depthmap = gr.Checkbox(label='Save depth map', value=True)
        return    [save_depthmap]

    def run(self,p,save_depthmap):
        def create_depth_mask_from_depth_map(img,save_depthmap):
            img = copy.deepcopy(img.convert("RGBA"))
            if save_depthmap:
                images.save_image(img, p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, p=p)
            mask_img = copy.deepcopy(img.convert("L"))
            mask_datas = mask_img.getdata()
            datas = img.getdata()
            newData = []
            for i in range(len(mask_datas)):
                newData.append((datas[i][0],datas[i][1],datas[i][2],255-mask_datas[i]))
            img.putdata(newData)
            if save_depthmap:
                images.save_image(img, p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, p=p)
            return img

        sdmg = module_from_file("depthmap_for_depth2img",'extensions/multi-subject-render/scripts/depthmap_for_depth2img.py')
        sdmg = sdmg.SimpleDepthMapGenerator() #import midas
        d_m = sdmg.calculate_depth_maps(p.init_images[0],p.width,p.height)
        d_m = cut_depth_mask(d_m,save_depthmap)
        p.image_mask = d_m
        proc = process_images(p)
        return proc
