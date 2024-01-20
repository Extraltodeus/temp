from copy import deepcopy
import torch


class latent_zoom:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "LATENT": ("LATENT", {"forceInput": True}),
                "zoom": ("INT", {"default": 0, "min": -1000,"max": 1000,"step": 1})
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("LATENT",)
    CATEGORY = "latent"

    def replicate_outer(self, tensor, X):
        top_row = tensor[:, 0:1, :]
        bottom_row = tensor[:, -1:, :]
        top_replicated = top_row.repeat(1, X, 1)
        bottom_replicated = bottom_row.repeat(1, X, 1)
        tensor_with_extended_rows = torch.cat([top_replicated, tensor, bottom_replicated], dim=1)
        left_column_extended = tensor_with_extended_rows[:, :, 0:1]
        right_column_extended = tensor_with_extended_rows[:, :, -1:]
        left_replicated = left_column_extended.repeat(1, 1, X)
        right_replicated = right_column_extended.repeat(1, 1, X)
        tensor_with_extended_rows_and_columns = torch.cat([left_replicated, tensor_with_extended_rows, right_replicated], dim=2)
        return tensor_with_extended_rows_and_columns

    def simple_output(self, LATENT,zoom):
        if zoom == 0:
            return (LATENT,)
        new_latents = deepcopy(LATENT)
        samples = []
        for index in range(LATENT['samples'].shape[0]):
            bigger_latent = self.replicate_outer(new_latents['samples'][index],abs(zoom))
            samples.append(bigger_latent)
        new_latents['samples'] = torch.stack(samples, dim=0)
        return (new_latents,)


NODE_CLASS_MAPPINGS = {
    "latent_zoom": latent_zoom,
}
