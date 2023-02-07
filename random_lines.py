from PIL import Image, ImageDraw
from math import sqrt, sin, cos, pi
from random import randint
import os

# line_amount = 40000
line_amount_proportion = 20
line_length_proportion = 0.04
average = False
line_width = 2
accentuation = 1.1

def most_frequent_color(pixels):
    most_frequent_pixel = pixels[0]
    for count, color in pixels:
        if count > most_frequent_pixel[0]:
            most_frequent_pixel = (count, color)
    return most_frequent_pixel

def get_average_color(pixels):
    r = 0
    g = 0
    b = 0
    for f in range(0,len(pixels)-1):
        r+=pixels[f][0]
        g+=pixels[f][1]
        b+=pixels[f][2]
    r = int(r/len(pixels))
    g = int(g/len(pixels))
    b = int(b/len(pixels))
    return r,g,b

def get_distance(c1,c2):
    x3 = abs(c1[0]-c2[0])**2
    y3 = abs(c1[1]-c2[1])**2
    return sqrt(x3+y3)

def get_rand_coords():
    x1 = randint(0,x)
    y1 = randint(0,y)
    return x1,y1

def get_rand_coords2(c1):
    j = randint(0,360)/360
    x1 = sin(j*pi*2)*line_length+c1[0]
    y1 = cos(j*pi*2)*line_length+c1[1]
    if x1 < 1: x1 = 0
    if x1 > x: x1 = x-1
    if y1 < 1: y1 = 0
    if y1 > y: y1 = y-1
    return int(x1),int(y1)

def get_coords_in_between(c1,c2,p,distance):
    if c1[0] < c2[0]:
        x1 = c1[0]
        x2 = c2[0]
    else:
        x1 = c2[0]
        x2 = c1[0]
    if c1[1] < c2[1]:
        y1 = c1[1]
        y2 = c2[1]
    else:
        y1 = c2[1]
        y2 = c1[1]
    x3 = int (x - (x-x1) + (x2-x1)*1/distance*p)
    y3 = int (y - (y-y1) + (y2-y1)*1/distance*p)
    return x3,y3

def get_pixels_for_line(c1,c2):
    distance = int(get_distance(c1,c2))
    pixels = []
    coords_already_done = []
    for p in range(0,distance):
        coords_in_between = get_coords_in_between(c1,c2,p,distance)
        if coords_in_between not in coords_already_done:
            coords_already_done.append(coords_in_between)
            xy = coords_in_between
            if average :
                pixels.append( im.getpixel(coords_in_between) )
            else:
                pixels.append( coords_in_between )
    return pixels

def get_acc_val(c1):
    c = (c1-255/2)*accentuation+255/2
    if c<0:c=0
    if c>255:c=255
    return int(c)

def apply_accentuation(color):
    r = get_acc_val(color[0])
    g = get_acc_val(color[1])
    b = get_acc_val(color[2])
    return r,g,b

def main():
    global image
    draw = ImageDraw.Draw(image)
    for r in range(0,line_amount):
        c1 = get_rand_coords()
        c2 = get_rand_coords2(c1)
        pixels = get_pixels_for_line(c1,c2)
        try:
            if len(pixels) > 0:
                if average :
                    color = get_average_color(pixels)
                else:
                    color = im.getpixel(most_frequent_color(pixels))
                color = apply_accentuation(color)
                draw.line([c1,c2],fill=color,width = line_width)
        except Exception as e:
            pass
    image.save('./out.bmp')
    image.show()

def get_first_image_in_folder_that_is_not_a_bitmap():
    global im,x,y,image, line_length,line_amount
    types = [".png",".jpg",".jpeg"]
    files = os.listdir("./")
    for f in files :
        for type in types:
            if type in f:
                image_path = f
    im = Image.open(image_path)
    x, y = im.size
    diag = sqrt(x**2+y**2)
    line_length = int(diag*line_length_proportion)
    line_amount = int(diag*line_amount_proportion)
    print(f'{line_length*line_amount:,}')
    image = Image.new('RGB', (x,y))
    image.paste((0,0,0), (0,0,x,y))

if __name__ == '__main__':
    get_first_image_in_folder_that_is_not_a_bitmap()
    main()
