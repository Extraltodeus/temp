import torch
from comfy.samplers import calc_cond_batch, encode_model_conds
from comfy.sampler_helpers import convert_cond

class pre_cfg_perp_neg:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "clip":  ("CLIP",),
                                "neg_scale": ("FLOAT",   {"default": 1.0,  "min": 0.0, "max": 10.0,  "step": 1/10, "round": 0.01}),
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
            if len(args["conds_out"]) > 1:
                noise_pred_neg = args["conds_out"][1]
            else:
                return args["conds_out"]

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

selfnorm = lambda x: x / x.norm()
def normalize_adjust(a,b,strength=1):
    norm_a = torch.linalg.norm(a)
    a = selfnorm(a)
    b = selfnorm(b)
    res = b - a * (a * b).sum()
    if res.isnan().any():
        res = torch.nan_to_num(res, nan=0.0)
    a = a - res * strength
    return a * norm_a

class condDiffSharpeningNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "scale":     ("FLOAT",   {"default": 0.75, "min": 0.0, "max": 10.0, "step": 1/20, "round": 1/100}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model, scale):
        model_sampling = model.model.model_sampling
        sigma_max = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_max)).item()
        prev_cond   = None
        prev_uncond = None

        def sharpen_conds_pre_cfg(args):
            nonlocal prev_cond, prev_uncond
            conds_out = args["conds_out"]
            uncond = len(conds_out) > 1
            
            sigma  = args["sigma"][0].item()
            first_step = sigma > (sigma_max - 1)

            for b in range(len(conds_out[0])):
                for c in range(len(conds_out[0][b])):
                    if not first_step and sigma > 1:
                        if prev_cond is not None:
                            conds_out[0][b][c]   = normalize_adjust(conds_out[0][b][c], prev_cond[b][c], scale)
                        if prev_uncond is not None and uncond:
                            conds_out[1][b][c] = normalize_adjust(conds_out[1][b][c], prev_uncond[b][c], scale)

            prev_cond = conds_out[0]
            if uncond:
                prev_uncond = conds_out[1]

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(sharpen_conds_pre_cfg)
        return (m, )

def normalized_pow(t,p):
    t_norm = t.norm()
    t_sign = t.sign()
    t_pow  = (t / t_norm).abs().pow(p)
    t_pow  = selfnorm(t_pow) * t_norm * t_sign
    return t_pow

class condExpNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "exponent": ("FLOAT",   {"default": 0.8, "min": 0.0, "max": 10.0, "step": 1/20, "round": 1/100}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model, exponent):
        def exponentiate_conds_pre_cfg(args):
            if args["sigma"][0] <= 1: return args["conds_out"]

            conds_out = args["conds_out"]
            uncond = len(conds_out) > 1

            for b in range(len(conds_out[0])):
                conds_out[0][b] = normalized_pow(conds_out[0][b], exponent)
                if uncond:
                    conds_out[1][b] = normalized_pow(conds_out[1][b], exponent)

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(exponentiate_conds_pre_cfg)
        return (m, )

def topk_average(latent, top_k=0.25):
    max_values = torch.topk(latent, k=int(len(latent)*top_k), largest=True).values
    min_values = torch.topk(latent, k=int(len(latent)*top_k), largest=False).values
    max_val = torch.mean(max_values).item()
    min_val = torch.mean(torch.abs(min_values)).item()
    value_range = (max_val + min_val) / 2
    return value_range

class automatic_cfg_simplified:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model):
        def automatic_cfg(args):
            cond   = args["cond_denoised"]
            uncond = args["uncond_denoised"]
            x_orig = args["input"]

            if not torch.any(args['uncond_denoised']):
                return x_orig - cond

            cond_scale = args["cond_scale"]
            result = torch.zeros_like(x_orig)
            
            for b in range(len(x_orig)):
                for c in range(len(x_orig[b])):
                    mes = topk_average(8 * cond[b][c] - 7 * uncond[b][c])
                    result[b][c] = (x_orig[b][c] - uncond[b][c]) + ((x_orig[b][c] - cond[b][c]) - (x_orig[b][c] - uncond[b][c])) * 8 * (cond_scale / 10) / max(mes,0.01)

            return result

        m = model.clone()
        m.set_model_sampler_cfg_function(automatic_cfg)
        return (m, )