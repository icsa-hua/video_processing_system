import imageio
import numpy as np 
from PIL import Image 
from obs_system.conversion_module.interface.convertor import CameraProcessor
from scipy.ndimage import map_coordinates 
from tqdm import tqdm 
import os
import time
from datetime import datetime

class FrameConvertor(CameraProcessor): 
    def __init__(self,  frame, shape, model, output_path, model_name, max_workers=10, native_size=0, FOV=90,overlap=0.2):
        self.img = frame
        self.model = model
        self.output_path = output_path
        self.model_name = model_name
        self.native_size = native_size
        if self.model_name =="Yolo" or self.model_name=="YOLO": 
            self.native_size = 640 
        elif self.model_name =="maskrcnn" or self.model_name=="Mask R-CNN":
            self.native_size = 1024
        self.output_size = self.native_size
        
        self.FOV = FOV
        self.pano_array = self.img
        self.pano_width, self.pano_height, self.pano_dims = shape[1], shape[0], shape[2]
        self.cart_coordinates()
        
        self.overlap = overlap
        self.overlap_size = int(overlap*self.native_size) #to avoid losing information from the edges. 
        #Yolo -> 128 
        #Mask-r-cnn -> 204

    def cart_coordinates(self): 
        print("Cart Coordinates")
        W, H = self.output_size, self.output_size
        f = (0.5*W) / np.tan(np.radians(self.FOV)/2)
        u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        self.x = u - W /2
        self.y = H / 2 - v 
        self.z = f 

    def get_stride(self): 
        return self.native_size - self.overlap_size 

    def map_to_sphere(self, x, y, z, yaw_radian, pitch_radian): 
        print("Map to sphere")
        r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        theta = np.arccos(self.z/r)
        phi = np.arctan2(self.y,self.x)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta) 
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        theta_prime = np.arccos(sin_theta * sin_phi * np.sin(pitch_radian) + cos_theta * np.cos(pitch_radian))
        phi_prime = np.arctan2(sin_theta * sin_phi * np.cos(pitch_radian) - cos_theta * np.sin(pitch_radian), sin_theta * cos_phi)

        phi_prime += yaw_radian
        phi_prime = phi_prime % (2* np.pi)

        return theta_prime.flatten(), phi_prime.flatten()
    
    def sliding_windows(self, patch_size,overlap):
        print("Sliding Windows")
        H, W = self.img.shape[:2]
        stride = int(patch_size * (1-overlap))
        for y in range(0,H, stride):
            for x in range(0, W, stride):
                yield x, y, self.img[y:y + patch_size, x:x + patch_size]

    def video_conversion(self):
        print("Video Conversion")
        with imageio.get_writer(self.output_path, fps=30) as writer:
            for deg in tqdm(np.arange(0,360,0.25)):
                yaw_radian = np.radians(deg)
                pitch_radian = np.radians(90)
                theta, phi= self.map_to_sphere(self.x, self.y, self.z, yaw_radian, pitch_radian)
                U = phi * self.pano_width / (2 * np.pi)
                V = theta * self.pano_height / np.pi
                coords = np.vstack((V,U))
                colors = self.interpolate_color(coords)
                
                output_image = colors.reshape((self.output_size,self.output_size,3)).astype(np.uint8)
                writer.append_data(output_image)
    
    def interpolate_color(self, coords, method='bilinear'):
        print("Interpolate Colors")
        order = {'nearest' : 0, 'bilinear':1, 'bicubic':3}.get(method, 1)
        colors = np.stack([
            map_coordinates(self.img[:,:,i],coords, order=order, mode='reflect') for i in range(3)
        ], axis=-1)

        return colors

    def compute_patches(self):
        print("Compute patches")
        #Calculate how many patches are needed to cover the entire width and height of the original image, considering the overlap
        stride = self.native_size - self.overlap_size 
        num_patches_x = (self.pano_width - self.overlap_size) // stride
        num_patches_y = (self.pano_height - self.overlap_size) // stride
        
        if ((self.pano_width-self.overlap_size) % stride != 0):
            num_patches_x += 1 
        
        if ((self.pano_height-self.overlap_size) % stride != 0):
            num_patches_y += 1  

        return num_patches_x, num_patches_y, stride

    def decomposition(self): 
        print("Decomposition Initialized")
        patches = []
        positions = []
        num_patches_x, num_patches_y, stride = self.compute_patches()
        
        for ii in range(num_patches_y): #y -> x
            for jj in range(num_patches_x): #x-> y
        
                left = jj * stride
                top = ii * stride 

                if left + self.native_size > self.pano_array.shape[1]: 
                    left = self.pano_array.shape[1] - self.native_size 

                if top + self.native_size > self.pano_array.shape[0]: 
                    top = self.pano_array.shape[0] - self.native_size

                #Extract patch 
                patch = self.pano_array[top:top + self.native_size, left:left + self.native_size]
                patches.append(patch)
                positions.append((jj,ii))


        gen_path = './converted_mp4/generated_patches'
        gen_path =  os.path.abspath(gen_path)
        if not os.path.exists(gen_path): 
            os.makedirs(gen_path)

        for i, patch in enumerate(patches): 
            patch_img = Image.fromarray(patch)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            patch_img.save(f'{gen_path}/patch_{i}_{timestamp}.png')

        return patches, positions     

    def reconstruct_image(self,patches, positions):
        channel_count = patches[0].shape[2] if len(patches[0].shape) == 3 else 1
        stride = self.native_size - self.overlap_size
        # Using a floating point type to avoid overflow during accumulation
        full_image = np.zeros((self.pano_height, self.pano_width, channel_count), dtype=np.float32)
        contribution_counts = np.zeros((self.pano_height, self.pano_width, channel_count), dtype=np.float32)
        
        for (patch, (jj, ii)) in zip(patches, positions):
            left = jj * stride
            top = ii * stride
            
            # Correct handling of boundary conditions
            patch_height, patch_width, _ = patch.shape
            if left + patch_width > self.pano_width:
                left = self.pano_width - patch_width
            if top + patch_height > self.pano_height:
                top = self.pano_height - patch_height
            
            # Accumulate patch data and count contributions
            full_image[top:top + patch_height, left:left + patch_width] += patch
            contribution_counts[top:top + patch_height, left:left + patch_width] += 1
        
        # Avoid division by zero and normalize to get the final pixel values
        with np.errstate(divide='ignore', invalid='ignore'):
            full_image = np.divide(full_image, contribution_counts, where=(contribution_counts != 0))
            full_image[contribution_counts == 0] = 0  # Optionally set non-contributed regions to zero

        # Convert back to uint8 for image display
        # full_image = full_image.astype(np.uint8)
        
        return full_image

    

