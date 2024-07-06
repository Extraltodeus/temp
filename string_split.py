
class string_split_class:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_input": ("STRING", {"multiline": True, "forceInput": True}),
                "split_char": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0,"max": 100000000,"step": 1}),
            }
        }

    RETURN_TYPES = ("STRING","LIST",)
    FUNCTION = "exec"
    CATEGORY = "utils"

    def exec(self, text_input, split_char, seed):
        string_list = text_input.split(split_char)
        output = string_list[seed%len(string_list)]
        return (output,string_list,)
            
NODE_CLASS_MAPPINGS = {
    "string_split": string_split_class,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "string_split": "string_split_node",
}

