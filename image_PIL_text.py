from PIL import ImageDraw, ImageFont, ImageColor, Image
import numpy as np
import torch
import torchvision.transforms as transforms
import os
import qrcode
from copy import deepcopy

def PIL2IMG(PIL_image):
    images = []
    for x in range(len(PIL_image)):
        numpy_image = np.array(PIL_image[x])
        numpy_image = numpy_image / 255.0
        tensor_image = torch.from_numpy(numpy_image)
        tensor_image = tensor_image.unsqueeze(0)
        images.append(tensor_image)
    images_tensor = torch.cat(images, 0)
    return images_tensor

class image_PIL_text_class:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        fonts  = [f.split(".")[0] for f in os.listdir("./fonts") if ".ttf" in f]
        colors = [name for name, code in ImageColor.colormap.items()]
        return {
            "required": {
                "input_image": ("IMAGE",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "font_size":  ("INT", {"default": 64,"min": 1,"max": 1000,"step": 1}),
                "shadow_size":  ("INT", {"default": 1,"min": 0,"max": 8,"step": 1}),
                "shadow_color":  (['black','white'], {"default": "black"}),
                "x_pos":  ("INT", {"default": 20,"min": 0,"max": 10000,"step": 1}),
                "y_pos":  ("INT", {"default": 20,"min": 0,"max": 10000,"step": 1}),
                "text_color": (colors, {"default": "lime"}),
                "centered": (["no","yes"], {"default": "no"}),
                "enabled": ("INT", {"default": 1, "min": 0,"max": 1,"step": 1}),
                "font": (fonts,),
                "vertical_spacing": ("FLOAT", {"default": 1, "min": 0,"max": 2,"step": 0.05}),
                "from_bottom": ("BOOLEAN",{}),
            }
        }

    FUNCTION = "exec"
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "image"

    # def write_on_image(self, PIL_image_input, text, font_size, x_pos, y_pos, text_color, shadow_size, shadow_color, centered):
    #     draw = ImageDraw.Draw(PIL_image_input)
    #     font = ImageFont.truetype('Roboto-Regular.ttf', font_size)
    #     shadows = [-shadow_size,shadow_size,0]
    #     if centered:
    #         text_width, text_height = draw.textsize(text, font)
    #         x_pos -= text_width / 2
    #     for x_shift in shadows:
    #         for y_shift in shadows:
    #             draw.text((x_pos+x_shift, y_pos+y_shift), text, font=font, fill=shadow_color)
    #     draw.text((x_pos, y_pos), text, font=font, fill=text_color)
    #     return PIL_image_input

    def IGM2PIL(self, input_image):
        PIL_images = []
        for x in range(len(input_image)):
            i = 255. * input_image[x].cpu().numpy()
            image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            PIL_images.append(image)
        return PIL_images
    
    def PIL2IMG(self, PIL_image):
        images = []
        for x in range(len(PIL_image)):
            numpy_image = np.array(PIL_image[x])
            numpy_image = numpy_image / 255.0
            tensor_image = torch.from_numpy(numpy_image)
            tensor_image = tensor_image.unsqueeze(0)
            images.append(tensor_image)
        images_tensor = torch.cat(images, 0) # Concatenate all images into a single tensor along the batch dimension
        return images_tensor
    
    def write_on_image(self, PIL_image_input, text, font_size, x_pos, y_pos, text_color, shadow_size, shadow_color, centered, font,vertical_spacing,from_bottom):
        draw = ImageDraw.Draw(PIL_image_input)
        font = ImageFont.truetype(f'./fonts/{font}.ttf', font_size)
        shadows = [-shadow_size, shadow_size, 0]
        image_width, image_height = PIL_image_input.size

        # Function to draw text with shadows
        def draw_text_with_shadow(x, y, txt):
            for x_shift in shadows:
                for y_shift in shadows:
                    draw.text((x + x_shift, y + y_shift), txt, font=font, fill=shadow_color)
            draw.text((x, y), txt, font=font, fill=text_color)

        # Split the text into words
        # words = text.split()
        lines = [text]
        # lines = []
        # while words:
        #     line = ''
        #     if font.getsize(words[0])[0] > image_width - x_pos * 2:
        #         break
        #         # words[0:1] = list(words[0][i:i+image_width] for i in range(0, len(words[0]), image_width))
        #     while words and font.getsize(line + words[0])[0] <= image_width*2 - x_pos * 2:
        #         line += (words.pop(0) + ' ')
        #     lines.append(line)

        if from_bottom:
            lines = list(reversed(lines))
            text_width, text_height = draw.textsize(lines[0], font=font)
            y_pos-= text_height*vertical_spacing*2
            # y_pos-= text_height*vertical_spacing*len(lines)
        # Draw each line
        for line in lines:
            text_width, text_height = draw.textsize(line, font=font)
            if centered:
                line_x_pos = x_pos - text_width / 2
            else:
                line_x_pos = x_pos
            if from_bottom:
                draw_text_with_shadow(line_x_pos, image_height+y_pos, line)
                y_pos -= text_height*vertical_spacing
            else:
                draw_text_with_shadow(line_x_pos, y_pos, line)
                y_pos += text_height*vertical_spacing
        return PIL_image_input
    
    def exec(self, input_image, text, font_size, shadow_size, shadow_color, x_pos, y_pos, text_color, centered, enabled, font, vertical_spacing,from_bottom):
        if enabled != 1:
            return (input_image,)
        input_image = self.IGM2PIL(input_image)
        PIL_images = []
        for x in range(len(input_image)):
            image = self.write_on_image(input_image[x], text, font_size, x_pos, y_pos, text_color, shadow_size, shadow_color, centered=="yes", font,vertical_spacing,from_bottom)
            PIL_images.append(image)
        PIL_images = self.PIL2IMG(PIL_images)
        return (PIL_images,)


def generate_qr_code(resolution, url, border, e_c):
    err_corr_dict = {
        "L":qrcode.constants.ERROR_CORRECT_L,
        "M":qrcode.constants.ERROR_CORRECT_M,
        "Q":qrcode.constants.ERROR_CORRECT_Q,
        "H":qrcode.constants.ERROR_CORRECT_H,
    }
    err_corr = err_corr_dict[e_c]
    qr = qrcode.QRCode(
        version=1,
        error_correction=err_corr,
        box_size=resolution // 100,
        border=border,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    return img


class QRgenerator:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        error_corrections = ["L","M","Q","H"]
        return {
            "required": {
                "text": ("STRING", {}),
                "size":  ("INT", {"default": 512,"min": 0,"max": 10000,"step": 1}),
                "border":  ("INT", {"default": 1,"min": 0,"max": 100,"step": 1}),
                "error_correction": (error_corrections, {}),

            }
        }

    FUNCTION = "exec"
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "image"

    def exec(self, text, size, border, error_correction):
        qrcode_image = generate_qr_code(size,text,border,error_correction)
        qrcode_image = qrcode_image.convert('RGB')
        result_image = PIL2IMG([qrcode_image])
        return (result_image,)

def generate_log_spiral(image, width, height, line_thickness, line_resolution_pow, initial_radius, winding_factor, iter_shift, iter_shift_mult, rotation_shift):
    a, b = initial_radius, winding_factor
    cx, cy = width // 2, height // 2

    theta = np.linspace(0, 10 * np.pi, 10**line_resolution_pow)
    r = a * np.exp(b * theta)

    x = (cx + r * np.cos(theta + 2 * np.pi * (iter_shift + rotation_shift) * iter_shift_mult)).astype(int)
    y = (cy + r * np.sin(theta + 2 * np.pi * (iter_shift + rotation_shift) * iter_shift_mult)).astype(int)

    dx = np.arange(-line_thickness // 2, line_thickness // 2 + 1)
    dy = np.arange(-line_thickness // 2, line_thickness // 2 + 1)

    x_range = (x[:, None] + dx)
    y_range = (y[:, None] + dy)

    # x_valid = (x_range >= 0) & (x_range < width)
    # y_valid = (y_range >= 0) & (y_range < height)

    xx, yy = np.meshgrid(x_range.ravel(), y_range.ravel(), sparse=True)
    xx = xx.ravel()
    yy = yy.ravel()

    valid_indices = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
    image[yy[valid_indices], xx[valid_indices]] = 1

    return image

class spiral_generator:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width":  ("INT", {"default": 1024,"min": 0,"max": 10000,"step": 8}),
                "height":  ("INT", {"default": 1024,"min": 0,"max": 10000,"step": 8}),
                "iterations":  ("INT", {"default": 1,"min": 1,"max": 1000,"step": 1}),
                "iter_shift_mult":  ("FLOAT", {"default": 1, "min": 0,"max": 1,"step": 0.01}),
                "rotation_shift":  ("FLOAT", {"default": 0, "min": 0,"max": 1,"step": 0.001}),
                "thickness":  ("INT", {"default": 2,"min": 1,"max": 100,"step": 1}),
                "line_resolution_pow":  ("INT", {"default": 4,"min": 1,"max": 10,"step": 1}),
                "initial_radius": ("FLOAT", {"default": 2, "min": 0,"max": 100,"step": 0.01}),
                "winding_factor": ("FLOAT", {"default": 0.2, "min": 0,"max": 100,"step": 0.01}),
                "invert": ("BOOLEAN",{"default": False}),
            }
        }

    FUNCTION = "exec"
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "image"

    def exec(self, width, height, iterations, iter_shift_mult, rotation_shift, thickness, line_resolution_pow, initial_radius, winding_factor, invert):
        image = np.zeros((height, width))
        for i in range(iterations):
            iter_shift = 1/iterations*(i+1)
            image = generate_log_spiral(image, width, height, thickness, line_resolution_pow, initial_radius, winding_factor, iter_shift, iter_shift_mult, rotation_shift)
        
        image = np.stack((image, image, image), axis=-1).astype(np.uint8)
        tensor_image = torch.from_numpy(image)
        if invert: tensor_image=1-tensor_image
        tensor_image = tensor_image.unsqueeze(0)
        images_tensor = torch.cat([tensor_image], 0)
        return (images_tensor,)

def add_noise_based_on_darkness(image_tensor, exponent, noise_intensity, RGB_noise, darken):
    image_tensor = image_tensor.permute(2, 0, 1)
    brightness = torch.mean(image_tensor, dim=0)
    darkness = (1 - brightness/brightness.max().item())**exponent*noise_intensity
    noisy_image = deepcopy(image_tensor)
    noise = torch.rand_like(image_tensor)
    noise = torch.clamp(noise-darken, 0, 1)
    for i in range(len(image_tensor)):
        if RGB_noise:
            noise_layer = noise[i]
        else:
            noise_layer = noise[0]
        noisy_image[i] = image_tensor[i]*(1-darkness)+noise_layer*darkness
    noisy_image = torch.clamp(noisy_image, 0, 1)
    noisy_image = noisy_image.permute(1, 2, 0)
    return noisy_image

class realisticNoiseNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "exponent":  ("INT", {"default": 3,"min": 0,"max": 12,"step": 1}),
                "noise_intensity":  ("FLOAT", {"default": 0.16, "min": 0,"max": 1,"step": 0.01}),
                "darken":  ("FLOAT", {"default": 0, "min": 0,"max": 1,"step": 0.1}),
                "RGB_noise": ("BOOLEAN",{"default": True}),
                "enabled": ("BOOLEAN",{"default": True}),
            }
        }

    FUNCTION = "exec"
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "image"

    def exec(self, input_image, exponent, noise_intensity, darken, RGB_noise, enabled):
        if not enabled: return (input_image,)
        output_image = deepcopy(input_image)
        for x in range(len(input_image)):
            output_image[x] = add_noise_based_on_darkness(input_image[x],exponent,noise_intensity, RGB_noise, darken)
        return (output_image,)
    
NODE_CLASS_MAPPINGS = {
    "Write on PIL image": image_PIL_text_class,
    "QRcode image generator": QRgenerator,
    "Spiral image generator": spiral_generator,
    "Noise Proportional to darkness":realisticNoiseNode,
}
