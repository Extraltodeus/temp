from .nodes import *

NODE_CLASS_MAPPINGS = {
    "Pre CFG perp-neg": pre_cfg_perp_neg,
    "Pre CFG sharpening": condDiffSharpeningNode,
    "Pre CFG exponentiation": condExpNode,
    "Automatic CFG simplified": automatic_cfg_simplified
}
