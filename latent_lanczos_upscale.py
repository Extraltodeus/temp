from copy import deepcopy
import torch
import math

def sinc(x):
    if x == 0:
        return 1.0
    return torch.sin(math.pi * x) / (math.pi * x)

def lanczos(x, a=3):
    if abs(x) < 1e-5:
        return torch.tensor(1.0)
    if abs(x) > a:
        return torch.tensor(0.0)
    return sinc(x) * sinc(x/a)
    
def upscale_lanczos(tensor, scale_factor, a=3):
    # Input tensor: [C, H, W]
    C, H, W = tensor.shape
    
    new_H = int(math.ceil(H * scale_factor))
    new_W = int(math.ceil(W * scale_factor))
    
    # Generate the coordinates for the original and the upscaled images
    orig_coords_x = torch.linspace(0, W - 1, steps=W)
    orig_coords_y = torch.linspace(0, H - 1, steps=H)
    
    up_coords_x = torch.linspace(0, W - 1, steps=new_W)
    up_coords_y = torch.linspace(0, H - 1, steps=new_H)
    
    # Calculate the contributions of each pixel based on the lanczos kernel
    contributions_x = torch.stack([lanczos((up_x - orig_x) / scale_factor, a) for up_x in up_coords_x for orig_x in orig_coords_x])
    contributions_y = torch.stack([lanczos((up_y - orig_y) / scale_factor, a) for up_y in up_coords_y for orig_y in orig_coords_y])
    
    contributions_x = contributions_x.view(new_W, W)
    contributions_y = contributions_y.view(new_H, H)
    
    # Normalize contributions
    contributions_x /= contributions_x.sum(dim=1, keepdim=True)
    contributions_y /= contributions_y.sum(dim=1, keepdim=True)
    
    # Upscale width
    upscaled_tensor_w = torch.einsum('jw,chw->chj', contributions_x, tensor)
    
    # Upscale height
    upscaled_tensor_hw = torch.einsum('ih,chj->cij', contributions_y, upscaled_tensor_w)
    
    return upscaled_tensor_hw

class latent_lanczos_upscale_by:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "LATENT": ("LATENT", {"forceInput": True}),
                "upscale_by": ("FLOAT", {"default": 1, "min": 0,"max": 12,"step": 0.5}),
                "support": ("INT", {"default": 3, "min": 0,"max": 512,"step": 1}),
                "in_X_times": ("INT", {"default": 1, "min": 1,"max": 16,"step": 1})
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("LATENT",)
    CATEGORY = "latent"

    def simple_output(self, LATENT,upscale_by,support,in_X_times):
        if upscale_by == 1:
            return (LATENT,)
        new_latents = deepcopy(LATENT)
        samples = []
        for index in range(LATENT['samples'].shape[0]):
            if in_X_times == 1:
                bigger_latent = upscale_lanczos(new_latents['samples'][index],upscale_by,support)
            else:
                scale_factor_each_pass = upscale_by ** (1/in_X_times)
                bigger_latent = upscale_lanczos(new_latents['samples'][index],scale_factor_each_pass,support)
                for x in range(in_X_times-1):
                    bigger_latent = upscale_lanczos(bigger_latent,scale_factor_each_pass,support)
            samples.append(bigger_latent)
        new_latents['samples'] = torch.stack(samples, dim=0)
        return (new_latents,)

def upscale_latent_x2(tensor):
    height, width = tensor.shape
    upscaled_tensor = torch.zeros((height*2, width*2), dtype=tensor.dtype)
    
    upscaled_tensor[::2, ::2] = tensor
    upscaled_tensor[1::2, ::2] = tensor
    upscaled_tensor[::2, 1::2] = tensor
    upscaled_tensor[1::2, 1::2] = tensor
    return upscaled_tensor

def upscale_latent_x2(tensor):
    """
    Upscales a given 4-channel latent tensor by a factor of 2, replicating each pixel into a 2x2 square.
    :param tensor: A 3D tensor with shape (C, H, W) to be upscaled.
    :return: The upscaled 3D tensor with shape (C, H*2, W*2).
    """
    # Get the number of channels, and the original height and width
    channels, height, width = tensor.shape
    
    # Prepare the upscaled tensor with zeros
    upscaled_tensor = torch.zeros((channels, height*2, width*2), dtype=tensor.dtype)
    
    # Replicate each pixel into a 2x2 block for each channel
    for c in range(channels):
        upscaled_tensor[c, ::2, ::2] = tensor[c]
        upscaled_tensor[c, 1::2, ::2] = tensor[c]
        upscaled_tensor[c, ::2, 1::2] = tensor[c]
        upscaled_tensor[c, 1::2, 1::2] = tensor[c]
    
    return upscaled_tensor

def generate_upscale_mask(height, width):
    tensor = torch.zeros((2*height, 2*width), dtype=torch.float32)
    tensor[::2, ::2] = 1
    return tensor

class masked_quarter_upscale:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "LATENT": ("LATENT", {"forceInput": True}),
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("LATENT","MASK",)
    CATEGORY = "latent"

    def simple_output(self, LATENT):
        new_latents = deepcopy(LATENT)
        c, y, x = new_latents['samples'][0].shape
        masked_upscale = generate_upscale_mask(y, x)

        samples = []
        for index in range(len(new_latents['samples'])):

            bigger_latent = upscale_latent_x2(new_latents['samples'][index])
            samples.append(bigger_latent)
        
        new_latents['samples'] = torch.stack(samples, dim=0)
        return (new_latents, masked_upscale, )

NODE_CLASS_MAPPINGS = {
    "latent_lanczos_upscale_by": latent_lanczos_upscale_by,
    "latent_masked_upscale_by_two": masked_quarter_upscale,
}
