import imageio
import numpy as np 
from PIL import Image 
from scipy.ndimage import map_coordinates 
from tqdm import tqdm 
from concurrent.futures import ThreadPoolExecutor
import cv2 


class CameraProcessor(): 
    def __init__(self,  frame, model, output_path, max_workers=10, native_size=1024, FOV=90,overlap=0.2):
        self.img = frame
        self.model = model
        self.output_path = output_path
        self.output_size = native_size
        self.FOV = FOV
        self.pano_array = np.array(self.img) 
        self.pano_width, self.pano_height = self.img.size
        self.cart_coordinates()
        self.native_size = native_size
        self.overlap = overlap
        self.overlap_size = int(overlap*native_size) #to avoid losing information from the edges. 

    def cart_coordinates(self): 
        W, H = self.output_size
        f = (0.5*W) / np.tan(np.radians(self.FOV)/2)
        u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        self.x = u - W /2
        self.y = H / 2 - v 
        self.z = f 

    def map_to_sphere(self, x, y, z, yaw_radian, pitch_radian): 
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
        H, W = self.img.shape[:2]
        stride = int(patch_size * (1-overlap))
        for y in range(0,H, stride):
            for x in range(0, W, stride):
                yield x, y, self.img[y:y + patch_size, x:x + patch_size]

    def video_conversion(self, output_file):
        with imageio.get_writer(output_file, fps=30) as writer:
            for deg in tqdm(np.arange(0,360,0.25)):
                yaw_radian = np.radians(deg)
                pitch_radian = np.radians(90)
                theta, phi= self.map_to_sphere(self.x, self.y, self.z, yaw_radian, pitch_radian)
                U = phi * self.pano_width / (2 * np.pi)
                V = theta * self.pano_height / np.pi
                coords = np.vstack((V,U))
                colors = self.interpolate_color(coords)
                output_image = colors.reshape((self.output_size[1],self.output_size[0],3)).astype(np.uint8)
                writer.append_data(output_image)
    
    def interpolate_color(self, coords, method='bilinear'):
        
        order = {'nearest' : 0, 'bilinear':1, 'bicubic':3}.get(method, 1)
        colors = np.stack([
            map_coordinates(self.img[:,:,i],coords, order=order, mode='reflect') for i in range(3)
        ], axis=-1)

        return colors

    def compute_patches(self):
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
        patches = []
        num_patches_x, num_patches_y, stride = self.compute_patches()
        for ii in range(num_patches_y):
            for jj in range(num_patches_x): 
                left = jj * stride
                top = ii * stride 

                if left + self.native_size > self.pano_array.shape[1]: 
                    left = self.pano_array.shape[1] - self.native_size 

                if top + self.native_size > self.pano_array.shape[0]: 
                    top = self.pano_array.shape[0] - self.native_size

                #Extract patch 
                patch = self.pano_array[top:top + self.native_size, left:left + self.native_size]
                patches.append(patch)

        if not os.path.exists('generated_patches'): 
            gen_path = os.path.join(self.output_path,'/generated_patches')
            os.makedirs(gen_path)

        for i, patch in enumerate(patches): 
            patch_img = Image.fromarray(patch)
            patch_img.save(f'output_patches/patch_{i}.png')

        return patches    




    


# with imageio.get_writer("360video.mp4", fps=30) as writer: 
#     for deg in tqdm(np.arange(0,360,0.25)): 
#         yaw_radian = np.radians(deg) 
#         pitch_radian = np.radians(90)
#         theta, phi = map_to_sphere(x, y, z, yaw_radian, pitch_radian)
#         U = phi * pano_width / (2 * np.pi)
#         V = theta * pano_height / np.pi
#         coords = np.vstack((V,U))

#         colors = interpolate_color(coords, pano_array)
#         output_image = colors.reshape((H,W,3)).astype(np.uint8)
#         writer.append_data(output_image)



#Processing Large 360 IMages from Cameras 

# #1. Decomposing the Image: 
# def sliding_window(image,patch_size, overlap): 
#     stride = int(patch_size * (1-overlap))
#     for y in range(0,image.shape[0], stride):
#         for x in range(0,image.shape[1], stride):
#             yield x, y, image[y:y + patch_size, x:x + patch_size]

# #2. Multiple Processing 
# def process_patch(patch): 
#     return _inference_patch(patch)

# def parallel_patches(image, patch_size, overlap ):
#     patches = [image[y:y + patch_size, x:x + patch_size] for x,y in sliding_window(image,patch_size,overlap)]
#     with ThreadPoolExecutor(max_workers=4) as executor: 
#         results = list(executor.map(process_patch, patches))

#3. Recombining Results 
# def recombining_detections(patches_info, results, img_original_shape): 
#     detections = [] 
#     for (x_offset, y_offset, _), result in zip(patches_info, results): 
#         for bbox in result: 
#             adjusted_bbox = adjust_bbox(bbox, x_offset, y_offset)
#             detections.append(adjusted_bbox)
#     return non_maximum_suppression(detections)


#4. MQTT Broker 

# #--Setup Client 
# def on_connect(client, userdata, flags, rc): 
#     print(" Connected with results code " + str(rc))
#     client.subscribe("topic/test")

# client = mqtt.Client()
# client.on_connect = on_connect
# client.connect("192.168.1.100", 1883, 60)
# client.loop_start() 






























