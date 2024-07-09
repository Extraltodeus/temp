import torch
from comfy.samplers import calc_cond_batch, encode_model_conds
from comfy.sampler_helpers import convert_cond

class pre_cfg_perp_neg:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "clip":  ("CLIP",),
                                "neg_scale": ("FLOAT",   {"default": 1.0,  "min": 0.0, "max": 100.0,  "step": 1/10, "round": 0.01}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model, clip, neg_scale):

        empty_cond, pooled = clip.encode_from_tokens(clip.tokenize(""), return_pooled=True)
        nocond = convert_cond([[empty_cond, {"pooled_output": pooled}]])

        def pre_cfg_perp_neg_function(args):
            noise_pred_pos = args["conds_out"][0]
            noise_pred_neg = args["conds_out"][1]
            model_options = args["model_options"]
            timestep = args["timestep"]
            model = args["model"]
            x = args["input"]
            
            nocond_processed = encode_model_conds(model.extra_conds, nocond, x, x.device, "negative")
            (noise_pred_nocond,) = calc_cond_batch(model, [nocond_processed], x, timestep, model_options)

            pos = noise_pred_pos - noise_pred_nocond
            neg = noise_pred_neg - noise_pred_nocond

            perp = neg - ((torch.mul(neg, pos).sum())/(torch.norm(pos)**2)) * pos
            perp_neg = perp * neg_scale
            
            return [noise_pred_nocond + pos, noise_pred_nocond + perp_neg]

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_perp_neg_function)
        return (m, )
