import imageio 
import time 
import os

directory = '/home/icsa_jim/Jupyter Notebook/EDGEAI_OBS_ROI/Obstacle_Recognition_Edge_Ai-main/converted_mp4/det/images/'
image_paths = sorted(
    [f for f in os.listdir(directory) if f.endswith('.png')],
    key=lambda x: int(x.split('_')[-1].split('.')[0])
)
image_paths = [os.path.join(directory, path) for path in image_paths if path.endswith('.png')]

images = [imageio.imread(path) for path in image_paths]
imageio.mimsave('animated.gif', images)
# with imageio.get_writer('obs.gif', mode='I') as writer: 
#     for filename in os.listdir(directory): 
#         print(filename)
#         image = imageio.imread(filename) 
#         writer.append_data(image)

