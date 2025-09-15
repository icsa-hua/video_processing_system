from obs_system.utils.logger import logger 

import math 
import torch
import cv2 
import numpy as np 

from ultralytics import YOLO 


def make_tiles(W, H, tile=640, overlap=0.2):

    if overlap < 1.0: 
        sx = int(tile * (1 - overlap))
        sy = int(tile * (1 - overlap))
    else: 
        sx = tile - int(overlap) 
        sy = tile - int(overlap) 

    xs = list(range(0,max(1, W - tile + 1), sx))
    ys = list(range(0,max(1, H - tile + 1), sy))

    if xs[-1] != W-tile:
        xs.append(W-tile) 

    if ys[-1] != H-tile: 
        ys.append(H-tile) 

    for y0 in ys: 
        for x0 in xs: 
            yield x0, y0, tile, tile 


def letterbox(img, new_shape=(640,640)): 
    H, W = img.shape[:2] 

    if (H,W) == new_shape: return img, (1.0,1.0), (0,0) 

    radius = min(new_shape[0] / H, new_shape[1]/ W) 
    new_H, new_W = int(round(H * radius)), int(round(W * radius)) 

    img_resized = cv2.resize(img, (new_H, new_W), interpolation=cv2.INTER_LINEAR) 

    top_y = (new_shape[0] - new_H) // 2 
    left_x = (new_shape[1] - new_W) // 2 

    canvas = np.full((new_shape[0], new_shape[1], 3), 114, dtype=np.uint8) 
    canvas[top_y:top_y + new_H, left_x:left_x+new_W] = img_resized 
    
    sx, sy = (radius, radius)
    px, py = left_x, top_y 

    return canvas, (sx, sy), (px, py) 


# code based on https://github.com/collinswakholi/ImgTiler/blob/main/ImgTiler/ImageTiler.py
# Create Tiles from the original image. 
def check_image_type(image): 
    max_val,min_val = 255,0
    type_ = (str(image.dtype)).lower() 
    if "float" in type_: 
        max_val, min_val = 255, 0 
    elif 'uint8' in type_: 
        max_val, min_val = 255, 0 
    elif 'uint16' in type_: 
        max_val, min_val = 65535, 0 
    elif 'uint32' in type_: 
        max_val, min_val = 4294967295, 0 

    dtype = eval('np.'+type_) 
    return dtype, max_val, min_val 


def convert_from_uint(image, dtype=np.uint8, max_val=255, min_val=0): 

    type_ = (str(dtype)).lower() 
    if 'float' in type_:
        image = np.array(image/255*(max_val-min_val)+ min_val, dtype=dtype) 
    elif 'uint8' in type_: 
        image = np.array(image, dtype=np.uint8) 
    else: 
        image = np.array(round(image/255*(max_val-min_val)+min_val), dtype=dtype)
    return image 

    
def convert_to_uint(image, dtype=np.uint8, max_val=255, min_val=0):
    type_ = (str(dtype)).lower() 
    if 'uint' in type_: 
        image = np.array(image, dtype=np.uint8) 
    else:  
        image = np.array((image-min_val)/(max_val-min_val)*255, dtype=np.uint8) 

    return image 


def check_divisible(grid, image): 
    H, W = image.shape[:2]
    w_pad = (grid[1] - (W % grid[1])) % grid[1] 
    h_pad = (grid[0] - (H % grid[0])) % grid[0] 
    new_size = pad_image(image, (0,h_pad, 0, w_pad))
    return new_size 


def get_grid(image, tile=640, overlap:int|float=0.0): 
    stride = tile - overlap 
    rows = 1 if image.shape[0] <= tile else math.ceil((image.shape[0] - tile)/stride) + 1 
    cols = 1 if image.shape[1] <= tile else math.ceil((image.shape[1] - tile)/stride) + 1  
    return (rows, cols) 


def pad_image(image, padding): 
    if len(image.shape) == 2: 
        return np.pad(image, ((padding[0], padding[1]), (padding[2], padding[3])), 'constant', constant_values=0) 
    else: 
        return np.pad(image,  ((padding[0], padding[1]), (padding[2], padding[3]), (0,0)), 'constant', constant_values=0) 


def tile_coords(W, H, tile_size=640, overlap=0.15):

    if overlap < 1.0: 
        overlap = int(round(tile_size*overlap))

    stride = int(max(1, tile_size-overlap))

    xs = list(range(0,max(1, W - tile_size + 1), stride))
    ys = list(range(0,max(1, H - tile_size + 1), stride)) 

    if xs[-1] + tile_size < W: xs.append(W - tile_size)
    if ys[-1] + tile_size < H: ys.append(H - tile_size) 

    return xs, ys 


def extract_tile(image, x0, y0, tile_size=640):
    """
    Extract tile from the image, with 
    i (int): Row index of the tile 
    j (int): Column index of the tile 
    """
    if len(image.shape)==3: 
        tile = image[y0:y0+tile_size, x0:x0+tile_size, :]
    else: 
        tile = image[y0:y0+tile_size, x0:x0+tile_size] 
    return tile 


def split_image(image,frame_id, tile_size=640, show_rect=False , show_tiles=False, overlap=0.15): 
    if overlap < 1.0:  
        overlap = int(tile_size * overlap)

    grid = get_grid(image, tile=tile_size, overlap=overlap) 
    image = check_divisible(grid=grid, image=image) 
    dtype, max_val, min_val = check_image_type(image) 

    H, W = image.shape[:2]
    orig_H, orig_W = H, W

    tile_height, tile_width = tile_size , tile_size

    pad_top = max(0, tile_height - H) 
    pad_left = max(0, tile_width - W) 

    gain = 1 if tile_size == 640 else int(min(H/tile_size, W/tile_size)) 

    if pad_top !=0 or pad_left!=0: 
        image = pad_image(image, padding=(overlap, overlap, overlap))
        H, W = image.shape[:2] 

    c = image.shape[2] if len(image.shape) == 3 else 1 

    tiles = [] 
    xs, ys = tile_coords(W, H, tile_size=640, overlap=overlap) 

    for x0 in xs: 
        for y0 in ys: 
            tile = extract_tile(image,x0=x0, y0=y0, tile_size=tile_size) 
            meta = {
                    'frame_id':frame_id, 
                    'grid':grid, 
                    'max_value':max_val, 
                    'min_value':min_val, 
                    'top_y':y0, 
                    'left_x': x0, 
                    'pad_y':pad_top, 
                    'pad_x':pad_left, 
                    'gain': gain, 
                    't_wh': (tile_height, tile_width), 
                    'f_wh': (orig_H, orig_W),
                    'overlap':overlap
            }

            if c <= 3: 
                if show_rect: 
                    cv2.rectangle(tile, (overlap, overlap), (tile_width + overlap, tile_height + overlap), (255, 0, 0), 1) 
                    
            tile = convert_from_uint(tile, dtype=dtype, min_val=min_val, max_val=max_val) 
            tiles.append((tile, meta)) 

    draw_tiles(tiles, grid, dtype, min_val, max_val) if show_tiles else None 
    
    return tiles


def draw_tiles(tiles, grid, dtype, min_val, max_val): 
    import matplotlib.pyplot as plt 
    fig, ax = plt.subplots(grid[0], grid[1], figsize=(20,20)) 

    if np.any(np.array(grid)==1): ax = ax.flatten() 

    for i, tile in enumerate(tiles): 
        tile = convert_to_uint(tile, dtype=dtype, min_val=min_val, max_val=max_val)
        c = tiles[0].shape[2] if len(tiles[0].shape)==3 else 1 

        if c > 3: 
            tile = tile[:,:,0]
            plt.set_cmap('gray') 
        else : 
            tile = tile[:,:,::-1] if len(tile.shape) ==3 else tile 
            plt.set_cmap('gray') if len(tiles[0].shape) == 2 else None 

        if np.any(np.array(grid)==1): 
            ax[i].imshow(tile) 
            ax[i].set_title('Tile {}'.format(i+1)) 

        else: 
            ax[i //grid[1], i % grid[1]].imshow(tile) 
            ax[i //grid[1], i % grid[1]].set_title('Tile {}'.format(i + 1)) 
    plt.show() 


