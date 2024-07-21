import torch
import comfy.model_patcher
from comfy.samplers import calc_cond_batch, encode_model_conds
from comfy.sampler_helpers import convert_cond
from nodes import ConditioningConcat, ConditioningSetTimestepRange
from copy import deepcopy

ConditioningConcat = ConditioningConcat()
ConditioningSetTimestepRange = ConditioningSetTimestepRange()

weighted_average = lambda tensor1, tensor2, weight1: (weight1 * tensor1 + (1 - weight1) * tensor2)
selfnorm = lambda x: x / x.norm()

class pre_cfg_perp_neg:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "clip":  ("CLIP",),
                                "neg_scale": ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 10.0,  "step": 1/10, "round": 0.01}),
                                "context_length": ("INT", {"default": 1,  "min": 1, "max": 100,  "step": 1}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, clip, neg_scale, context_length):
        empty_cond, pooled = clip.encode_from_tokens(clip.tokenize(""), return_pooled=True)
        nocond = [[empty_cond, {"pooled_output": pooled}]]
        if context_length > 1:
            short_nocond = deepcopy(nocond)
            for x in range(context_length - 1):
                (nocond,) = ConditioningConcat.concat(nocond, short_nocond)
        nocond = convert_cond(nocond)

        @torch.no_grad()
        def pre_cfg_perp_neg_function(args):
            conds_out = args["conds_out"]
            noise_pred_pos = conds_out[0]

            if torch.any(conds_out[1]):
                noise_pred_neg = conds_out[1]
            else:
                return conds_out

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

            conds_out[0] = noise_pred_nocond + pos
            conds_out[1] = noise_pred_nocond + perp_neg

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_perp_neg_function)
        return (m, )

class pre_cfg_generate_negative:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "clip":  ("CLIP",),
                                "prompt": ("STRING", {"multiline": True}),
                                "neg_scale": ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 10.0,  "step": 1/20, "round": 0.01}),
                                "end_at_sigma": ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 100.0,  "step": 1/20, "round": 0.01}),
                                "context_length": ("INT", {"default": 1,  "min": 1, "max": 100,  "step": 1}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, clip, prompt, neg_scale, end_at_sigma, context_length):
        neg_cond, pooled = clip.encode_from_tokens(clip.tokenize(prompt), return_pooled=True)
        neg_cond = [[neg_cond, {"pooled_output": pooled}]]
        n_cond_slices = neg_cond[0][0].shape[1] // 77
        if context_length > n_cond_slices:
            empty_cond, pooled = clip.encode_from_tokens(clip.tokenize(""), return_pooled=True)
            empty_cond = [[empty_cond, {"pooled_output": pooled}]]
            short_empty_cond = deepcopy(empty_cond)
            for x in range(context_length - n_cond_slices):
                (neg_cond,) = ConditioningConcat.concat(neg_cond, short_empty_cond)
        neg_cond = convert_cond(neg_cond)

        @torch.no_grad()
        def pre_cfg_gen_neg_function(args):
            model_options = args["model_options"]
            conds_out = args["conds_out"]
            timestep  = args["timestep"]
            model  = args["model"]
            sigma  = args["sigma"]
            if sigma[0] <= end_at_sigma:
                return conds_out
            x_orig = args["input"]

            # x_cond = conds_out[0] / sigma[0]
            x_cond = conds_out[0] / (((sigma ** 2 + 1.0) ** 0.5) / (sigma))

            # x_cond = ((x_orig - (x_orig - cond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)

            neg_cond_processed = encode_model_conds(model.extra_conds, neg_cond, x_cond, x_orig.device, "negative")
            # neg_cond_processed = encode_model_conds(model.extra_conds, neg_cond, x_cond, x_orig.device)
            
            (noise_pred_neg_cond,) = calc_cond_batch(model, [neg_cond_processed], x_cond, timestep, model_options)


            conds_out[1] = noise_pred_neg_cond * neg_scale
            # / noise_pred_neg_cond.norm() * conds_out[0].norm()

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_gen_neg_function)
        return (m, )

@torch.no_grad()
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
                                "scale": ("FLOAT",   {"default": 0.75, "min": -10.0, "max": 10.0, "step": 1/20, "round": 1/100}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, scale):
        model_sampling = model.model.model_sampling
        sigma_max = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_max)).item()
        prev_cond   = None
        prev_uncond = None

        @torch.no_grad()
        def sharpen_conds_pre_cfg(args):
            nonlocal prev_cond, prev_uncond
            conds_out = args["conds_out"]
            uncond = torch.any(conds_out[1])
            
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

@torch.no_grad()
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

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, exponent):
        @torch.no_grad()
        def exponentiate_conds_pre_cfg(args):
            if args["sigma"][0] <= 1: return args["conds_out"]

            conds_out = args["conds_out"]
            uncond = torch.any(conds_out[1])

            for b in range(len(conds_out[0])):
                conds_out[0][b] = normalized_pow(conds_out[0][b], exponent)
                if uncond:
                    conds_out[1][b] = normalized_pow(conds_out[1][b], exponent)

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(exponentiate_conds_pre_cfg)
        return (m, )

@torch.no_grad()
def topk_average(latent, top_k=0.25):
    max_values = torch.topk(latent, k=int(len(latent)*top_k), largest=True).values
    min_values = torch.topk(latent, k=int(len(latent)*top_k), largest=False).values
    max_val = torch.mean(max_values).item()
    min_val = torch.mean(torch.abs(min_values)).item()
    value_range = (max_val + min_val) / 2
    return value_range

class automatic_pre_cfg:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "mode": (["automatic_cfg","strict_scaling","strict_scaling_average","strict_scaling_smallest","strict_scaling_biggest"],),
                                "sqrt_scale" : ("BOOLEAN", {"default": False}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, mode, sqrt_scale = False):
        @torch.no_grad()
        def automatic_pre_cfg(args):
            conds_out = args["conds_out"]
            uncond = torch.any(conds_out[1])
            cond_scale = args["cond_scale"]
            
            if not uncond:
                return conds_out

            for b in range(len(conds_out[0])):
                chans = []
                for c in range(len(conds_out[0][b])):
                    mes = topk_average(8 * conds_out[0][b][c] - 7 * conds_out[1][b][c])

                    if mode == "automatic_cfg":
                        new_scale = 0.8 / max(mes,0.01)
                    else:
                        new_scale = cond_scale / max(mes * 8, 0.01)
                    
                    if sqrt_scale:
                        new_scale = new_scale ** 0.5 #variation dampener

                    if "strict" in mode:
                        new_scale *= 0.8 #better after sqrt

                    if mode not in ["automatic_cfg","strict_scaling"]:
                        chans.append(new_scale)
                    else:
                        conds_out[0][b][c] *= new_scale
                        conds_out[1][b][c] *= new_scale

                if mode == "strict_scaling_average":
                    conds_out[0][b] *= sum(chans) / len(chans)
                    conds_out[1][b] *= sum(chans) / len(chans)
                elif mode == "strict_scaling_smallest":
                    conds_out[0][b] *= min(chans)
                    conds_out[1][b] *= min(chans)
                elif mode == "strict_scaling_biggest":
                    conds_out[0][b] *= max(chans)
                    conds_out[1][b] *= max(chans)

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(automatic_pre_cfg)
        return (m, )

class channel_multiplier_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "channel_1": ("FLOAT",   {"default": 1, "min": -10.0, "max": 10.0, "step": 1/100, "round": 1/100}),
                                "channel_2": ("FLOAT",   {"default": 1, "min": -10.0, "max": 10.0, "step": 1/100, "round": 1/100}),
                                "channel_3": ("FLOAT",   {"default": 1, "min": -10.0, "max": 10.0, "step": 1/100, "round": 1/100}),
                                "channel_4": ("FLOAT",   {"default": 1, "min": -10.0, "max": 10.0, "step": 1/100, "round": 1/100}),
                                "selection": (["both","cond","uncond"],),
                                "start_at_sigma": ("FLOAT", {"default": 15.0,  "min": 0.0, "max": 100.0,  "step": 1/100, "round": 1/100}),
                                "end_at_sigma":   ("FLOAT", {"default": 01.0,  "min": 0.0, "max": 100.0,  "step": 1/100, "round": 1/100}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, channel_1, channel_2, channel_3, channel_4, selection, start_at_sigma, end_at_sigma):
        chans = [channel_1, channel_2, channel_3, channel_4]
        @torch.no_grad()
        def channel_multiplier_function(args):
            conds_out = args["conds_out"]
            uncond = torch.any(conds_out[1])
            sigma  = args["sigma"]
            if sigma[0] <= end_at_sigma or sigma[0] > start_at_sigma:
                return conds_out
            for b in range(len(conds_out[0])):
                for c in range(len(conds_out[0][b])):
                    if selection in ["both","cond"]:
                        conds_out[0][b][c] *= chans[c]
                    if uncond and selection in ["both","uncond"]:
                        conds_out[1][b][c] *= chans[c]
            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(channel_multiplier_function)
        return (m, )

class support_empty_uncond_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "method": (["divide by CFG","from cond"],),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, method):
        @torch.no_grad()
        def support_empty_uncond(args):
            conds_out = args["conds_out"]
            uncond = torch.any(conds_out[1])
            cond_scale = args["cond_scale"]

            if not uncond and cond_scale > 1:
                if method == "divide by CFG":
                    conds_out[0] /= cond_scale
                else:
                    conds_out[1]  = conds_out[0]
            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(support_empty_uncond)
        return (m, )

def replace_timestep(cond):
    cond = deepcopy(cond)
    cond[0]['timestep_start'] = 999999999.9
    cond[0]['timestep_end']   = 0.0
    return cond

class zero_input_eight_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "do_on": (["uncond","cond"],),
                             "attention": (["both","self","cross"],),
                             "start_at_sigma": ("FLOAT", {"default": 15.0, "min": 0.0,  "max": 1000.0, "step": 1/100, "round": 1/100}),
                             "end_at_sigma":   ("FLOAT", {"default": 0.0,  "min": 0.0,  "max": 1000.0, "step": 1/100, "round": 1/100}),
                             "mix_scale":      ("FLOAT", {"default": 1.0,  "min": -2.0, "max": 2.0,    "step": 1/20,  "round": 1/100}),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, do_on, attention, start_at_sigma, end_at_sigma, mix_scale):
        cond_index = 1 if do_on == "uncond" else 0
        attn = {"both":["attn1","attn2"],"self":["attn1"],"cross":["attn2"]}[attention]

        def zero_input_eight_attention(q, k, v, extra_options, mask=None):
            return torch.zeros_like(q)

        @torch.no_grad()
        def zero_input_eight(args):
            conds_out = args["conds_out"]
            sigma = args["sigma"][0].item()

            if sigma > start_at_sigma or sigma <= end_at_sigma:
                return conds_out

            conds = args["conds"]
            cond_to_process = conds[cond_index]
            cond_generated  = torch.any(conds_out[cond_index])

            print("-"*40)
            print(cond_to_process[0]['timestep_start'])
            print(cond_to_process[0]['timestep_end'])
            print(sigma)
            print(args['timestep'][0].item())

            if not cond_generated:
                cond_to_process = replace_timestep(cond_to_process)
            elif mix_scale == 1:
                print(" Mix scale at one!\nPrediction generated for nothing.\nUse the node ConditioningSetTimestepRange to avoid generating if you want to use the full result.")

            model_options = deepcopy(args["model_options"])
            for att in attn:
                model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, zero_input_eight_attention, att, "input", 8)

            (noise_pred,) = calc_cond_batch(args['model'], [cond_to_process], args['input'], args['timestep'], model_options)

            if mix_scale == 1 or not cond_generated:
                conds_out[cond_index] = noise_pred
            elif cond_generated:
                conds_out[cond_index] = weighted_average(noise_pred,conds_out[cond_index],mix_scale)

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(zero_input_eight)
        return (m, )

@torch.no_grad()
def euclidean_weights(tensors,exponent=2,proximity_exponent=1,min_score_into_zeros=0):
    divider = tensors.shape[0]
    if exponent == 0:
        exponent = 6.55
    device = tensors.device
    distance_weights = torch.zeros_like(tensors).to(device = device)

    for i in range(len(tensors)):
        for j in range(len(tensors)):
            if i == j: continue
            current_distance = (tensors[i] - tensors[j]).abs() / divider
            if proximity_exponent > 1:
                current_distance = current_distance ** proximity_exponent
            distance_weights[i] += current_distance

    min_stack, _ = torch.min(distance_weights, dim=0)
    max_stack, _ = torch.max(distance_weights, dim=0)
    max_stack    = torch.where(max_stack == 0, torch.tensor(1), max_stack)
    sum_of_weights = torch.zeros_like(tensors[0]).to(device = device)

    max_stack -= min_stack

    for i in range(len(tensors)):
        distance_weights[i] -= min_stack
        distance_weights[i] /= max_stack
        distance_weights[i]  = 1 - distance_weights[i]
        distance_weights[i]  = torch.clamp(distance_weights[i], min=0)
        if min_score_into_zeros > 0:
            distance_weights[i]  = torch.where(distance_weights[i] < min_score_into_zeros, torch.zeros_like(distance_weights[i]), distance_weights[i])
        
        if exponent > 1:
            distance_weights[i] = distance_weights[i] ** exponent

        sum_of_weights += distance_weights[i]
    
    mean_score = (sum_of_weights.mean() / divider) ** exponent

    sum_of_weights = torch.where(sum_of_weights == 0, torch.zeros_like(sum_of_weights) + 1 / divider, sum_of_weights)
    result = torch.zeros_like(tensors[0]).to(device = device)

    for i in range(len(tensors)):
        distance_weights[i] /= sum_of_weights
        distance_weights[i]  = torch.where(torch.isnan(distance_weights[i]) | torch.isinf(distance_weights[i]), torch.zeros_like(distance_weights[i]), distance_weights[i])
        result = result + tensors[i] * distance_weights[i]

    return result, mean_score

class condConsensusSharpeningNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "scale": ("FLOAT",   {"default": 0.75, "min": -10.0, "max": 10.0, "step": 1/20, "round": 1/100}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, scale):
        model_sampling = model.model.model_sampling
        sigma_max = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_max)).item()
        prev_conds   = []
        prev_unconds = []

        @torch.no_grad()
        def sharpen_conds_pre_cfg(args):
            nonlocal prev_conds, prev_unconds
            conds_out = args["conds_out"]
            uncond = torch.any(conds_out[1])
            
            sigma  = args["sigma"][0].item()

            if sigma <= 1:
                return conds_out

            first_step = sigma > (sigma_max - 1)
            if first_step:
                prev_conds   = []
                prev_unconds = []

            prev_conds.append(conds_out[0] / conds_out[0].norm())
            if uncond:
                prev_unconds.append(conds_out[1] / conds_out[1].norm())

            if len(prev_conds) > 3:
                consensus_cond, mean_score_cond = euclidean_weights(torch.stack(prev_conds))
                consensus_cond = consensus_cond * conds_out[0].norm()
            if len(prev_unconds) > 3 and uncond:
                consensus_uncond, mean_score_uncond = euclidean_weights(torch.stack(prev_unconds))
                consensus_uncond = consensus_uncond * conds_out[1].norm()

            for b in range(len(conds_out[0])):
                for c in range(len(conds_out[0][b])):
                        if len(prev_conds) > 3:
                            conds_out[0][b][c] = normalize_adjust(conds_out[0][b][c], consensus_cond[b][c],  mean_score_cond * scale)
                        if len(prev_unconds) > 3 and uncond:
                            conds_out[1][b][c] = normalize_adjust(conds_out[1][b][c], consensus_uncond[b][c], mean_score_uncond * scale)

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(sharpen_conds_pre_cfg)
        return (m, )