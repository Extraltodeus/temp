import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from time import sleep
from math import pi, sin, ceil
from copy import deepcopy


input_folder=r'''
X:\sdout\input_images_for_vid
'''
input_folder=input_folder.replace("\n","")
# IPD = 8.4
# IPD = 9.3
# IPD = 12
# IPD = IPD/1.6
# MONITOR_W = 38.5
init_deviation = 42
init_shift = 0.5
power = pi/1.7
max_size = 960
overwrite = True
resize_for_estimation = False
save_depthmaps = True
lower_cut_treshold = -0.3
upper_cut_treshold = 1
upper_depth_output = 0.86
use_blur = True
x_blur = 3
y_blur = 3


init_deviation = init_deviation / upper_depth_output
file_format = [".png",".jpg",".jpeg",".bmp"]

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def write_depth(depth, bits=2, reverse=True):
    depth_min = depth.min()
    depth_max = depth.max()
    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0
    if not reverse:
        out = max_val - out

    if bits == 2:
        depth_map = out.astype("uint16")
    else:
        depth_map = out.astype("uint8")

    return depth_map

def remap_range(value, minIn, MaxIn, minOut, maxOut):
    if value > MaxIn: value = MaxIn;
    if value < minIn: value = minIn;
    finalValue = ((value - minIn) / (MaxIn - minIn)) * (maxOut - minOut) + minOut;
    return finalValue;

def generate_stereo(left_img, depth, reverse=False, init_shift=0.5, power=2):
    h, w, c = left_img.shape
    left_img = np.fliplr(left_img) if reverse else left_img
    depth = np.fliplr(depth) if reverse else depth

    # if reverse: depth = depth*-1
    depth_min = depth.min()
    depth_max = depth.max()
    depth = (depth - depth_min) / (depth_max - depth_min)

    right = np.zeros_like(left_img)

    # deviation_cm = IPD * 0.12
    # deviation = deviation_cm * MONITOR_W * (w / 1920)
    deviation = init_deviation * w / 768
    # print("\ndeviation:", deviation)
    # left_right = 1 if reverse else -1
    for row in range(h):
        for col in range(w):
            col_r = col + int((init_shift - remap_range(depth[row][col],lower_cut_treshold,upper_cut_treshold,0,upper_depth_output) ** power) * deviation)
            if col_r >= 0 and col_r < w :
                right[row][col_r] = left_img[row][col]

    right_fix = np.array(right)
    gray = cv2.cvtColor(right_fix, cv2.COLOR_BGR2GRAY)
    rows, cols = np.where(gray == 0)
    for row, col in zip(rows, cols):
        for offset in range(1, int(deviation)):
            r_offset = col + offset
            l_offset = col - offset
            if r_offset < w and not np.all(right_fix[row][r_offset] == 0):
                right_fix[row][col] = right_fix[row][r_offset]
                break
            if l_offset >= 0 and not np.all(right_fix[row][l_offset] == 0):
                right_fix[row][col] = right_fix[row][l_offset]
                break
    right_fix = np.fliplr(right_fix) if reverse else right_fix
    # print(min(col_shifts),max(col_shifts), w, reverse, col_r)
    return right_fix


def overlap(im1, im2):
    width1 = im1.shape[1]
    height1 = im1.shape[0]
    width2 = im2.shape[1]
    height2 = im2.shape[0]

    # final image
    composite = np.zeros((height2, width2, 3), np.uint8)

    # iterate through "left" image, filling in red values of final image
    for i in range(height1):
        for j in range(width1):
            try:
                composite[i, j, 2] = im1[i, j, 2]
            except IndexError:
                pass

    # iterate through "right" image, filling in blue/green values of final image
    for i in range(height2):
        for j in range(width2):
            try:
                composite[i, j, 1] = im2[i, j, 1]
                composite[i, j, 0] = im2[i, j, 0]
            except IndexError:
                pass

    return composite

def check_size(left_img):
    height, width = left_img.shape[:2]
    if height > max_size or width > max_size:
        if height >= width:
            ratio = max_size / height
            height = max_size
            width = ceil(width * ratio)
        else:
            ratio = max_size / width
            width = max_size
            height = ceil(height * ratio)
        left_img = cv2.resize(left_img, (width, height))
    return left_img

def run(model_path):
    """
    Run MonoDepthNN to compute depth maps.
    """
    # Input images
    img_list = os.listdir(input_folder)
    img_list.sort()

    # output dir
    output_dir = os.path.join(input_folder,"3d/")
    depthmap_output_dir = os.path.join(output_dir,"depthmaps/")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(depthmap_output_dir, exist_ok=True)
    outdir_list = os.listdir(output_dir)
    depthmap_outdir_list = os.listdir(depthmap_output_dir)

    # set torch options
    torch.cuda.empty_cache()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # select device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device: %s" % device)

    # load network
    # :
    model = DPTDepthModel(
        path=model_path,
        backbone="beitl16_512" if model_path.endswith("dpt_beit_large_512.pt") else "vitb_rn50_384",
        non_negative=True,
    )

    model.to(device)
    model.eval()

    for idx in tqdm(range(len(img_list))):
        try:
            sample = img_list[idx]
            if 'MiDaS_{}.png'.format(sample.split('.')[0]) not in depthmap_outdir_list and any(ff in sample for ff in file_format):
                # print(sample)
                left_img = cv2.imread(os.path.join(input_folder, sample))
                left_img = check_size(left_img)


                img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB) / 255.0

                transform = Compose(
                    [
                        Resize(
                            384 if resize_for_estimation else left_img.shape[:2][0],
                            384 if resize_for_estimation else left_img.shape[:2][1],
                            resize_target=None,
                            keep_aspect_ratio=True,
                            ensure_multiple_of=32,
                            resize_method="minimal",
                            image_interpolation_method=cv2.INTER_CUBIC,
                        ),
                        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        PrepareForNet(),
                    ]
                )
                #  Apply transforms
                image = transform({"image": img})["image"]

                #  Predict and resize to original resolution
                with torch.no_grad():
                    image = torch.from_numpy(image).to(device).unsqueeze(0)
                    depth = model.forward(image)

                    depth = (
                    torch.nn.functional.interpolate(
                    depth.unsqueeze(1),
                    size=left_img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                    )

                    # depth = cv2.blur(depth, (3, 3))


                    # daemonizer(stereo_thread,depth,depthmap_output_dir,left_img,output_dir,sample)
                generated = True
            else:
                generated = False
                depth = ""
                left_img = ""

            pool.apply_async(stereo_thread, (depth, depthmap_output_dir, left_img, output_dir, sample, generated, outdir_list, depthmap_outdir_list))
            # stereo_thread(depth, depthmap_output_dir, left_img, output_dir, sample, generated, outdir_list, depthmap_outdir_list)
                    # cv2.imshow('depth map', depth_map)
                    # cv2.imshow('side by side', stereo)
                    # cv2.imshow("anaglyph", anaglyph)
                    # cv2.waitKey(0)
                    # anaglyph = overlap(left_img, right_img)

        except Exception as e:
            raise e

def stereo_thread(depth,depthmap_output_dir,left_img,output_dir,sample, generated, outdir_list, depthmap_outdir_list):
    if generated:
        depth_map = write_depth(depth, bits=2, reverse=True)
        if save_depthmaps:
            cv2.imwrite(os.path.join(depthmap_output_dir, 'MiDaS_{}.png'.format(sample.split('.')[0])), depth_map)
    try:
        if ('MiDaS_{}.png'.format(sample.split('.')[0]) not in outdir_list or overwrite) and any(ff in sample for ff in file_format):
        # if True:
            if not generated and 'MiDaS_{}.png'.format(sample.split('.')[0]) in depthmap_outdir_list:
                depth_map = load_image(os.path.join(depthmap_output_dir, 'MiDaS_{}.png'.format(sample.split('.')[0])))
                left_img  = cv2.imread(os.path.join(input_folder, sample))
                left_img = check_size(left_img)

            if use_blur:
                depth_map = cv2.blur(depth_map, (x_blur, y_blur))
            print("\n",sample)
            original_left = deepcopy(left_img)
            # shifts=[0,1]
            # powers=[2,pi*2]
            # for s in shifts:
            #     for p in powers:
            right_img = generate_stereo(original_left, depth_map, True,init_shift,power)
            left_img  = generate_stereo(original_left, depth_map, False, init_shift,power)
            stereo = np.hstack([left_img, right_img])
            cv2.imwrite(os.path.join(output_dir, 'MiDaS_{}.png'.format(sample.split('.')[0])), stereo)
    except Exception as e:
        print(e)
        raise e

if __name__ == "__main__":
    import torch
    from torch.backends import cudnn
    from torchvision.transforms import Compose
    import multiprocessing
    pool = multiprocessing.Pool(processes=8)
    # MODEL_PATH = "model-f46da743.pt"
    # MODEL_PATH = "midas_v21-f6b98070.pt"
    MODEL_PATH = "dpt_swin2_tiny_256.pt"
    MODEL_PATH = "D:\StableDiffusion\Autres\stereoimage\stereo-image-generation\dpt_hybrid-midas-501f0c75.pt"
    MODEL_PATH = "D:\StableDiffusion\Autres\stereoimage\stereo-image-generation\dpt_beit_large_512.pt"

    # compute depth maps
    run(MODEL_PATH)
    pool.close()
    pool.join()
