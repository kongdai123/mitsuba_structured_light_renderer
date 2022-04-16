import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')


import multiprocessing
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
from mitsuba.render import  Emitter
from mitsuba.render import  Scene
import multiprocessing
from mitsuba.core import Thread, Vector3f,Float
from mitsuba.core.xml import load_file
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render_torch, write_bitmap, render

from mitsuba.core import Bitmap, Struct, Thread
from mitsuba.core.xml import load_file


def toSRGB(img):
    
    #Convert colors in Linear RGB Color Space to sRGB monitor Color Space
    
    #ref https://github.com/cmu-ci-lab/mitsuba_clt/blob/e41a0f725ba338401efb2ca3bc4cd8690fe6d446/src/mtsgui/test_simdtonemap.cpp
    return np.where(img < 0.0031308, 12.92 * img ,1.055 * np.power(img, 1.0/2.4) - 0.055)

def load_scene_file(scene_path, scene_dir = None):
    if scene_dir != None:
        fileResolver.append(secne_dir)
    return load_file(scene_path)


    
class mitsuba_renderer:
    def __init__(self):
        self.dest_file = './output/result'
        self.dtype = 'float32'
    
    def load_scene(self, scene_path, scene_dir = None):
        self.scene = load_file(scene_path, scene_dir)
        self.params = traverse(self.scene)
        
        
        
    def render(self, scene= None, verbose = False, write_out = False):
        if scene is not None:
            self.scene = scene
        self.scene.sensors()[0].sampler().seed(0)
        self.scene.integrator().render(self.scene, self.scene.sensors()[0])

        
    def write_to_numpy(self, sRGB = False, pix_type = Bitmap.PixelFormat.RGB):
        film = self.scene.sensors()[0].film()
        bmp = film.bitmap(raw=True)

        # Get linear pixel values as a numpy array for further processing
        bmp_linear_rgb = bmp.convert(pix_type, Struct.Type.Float32, srgb_gamma=sRGB)
        image_np = np.array(bmp_linear_rgb)
        image_np -= image_np.min()


        #when using the float32 data type, Mitsuba renders color to Linear RGB color space
        #therefore colors may look off, need to tone map to the sRGB color space
        #sRGB color space is the standard for modern computer monitors and the web, https://en.wikipedia.org/wiki/SRGB
        #when using the uint8 data type, Mitsuba renders to the sRGB color space
        
#         if sRGB and self.dtype != 'uint8':
#             image_np = toSRGB(image_np)
        
        return image_np
    
    def configure_pattern(self, pattern_param, pattern):
        
        self.params[pattern_param + ".data"] = pattern.ravel()
        self.params[pattern_param + ".resolution"] = [pattern.shape[0], pattern.shape[1]]
        self.params.update()
        


    def render_scene_with_patterns(self, pattern_param, patterns, render_dim, sRGB = False):
        render_num = patterns.shape[0]
#         print((self.scene.sensors()[0].film().)
#         (self.scene.sensors()[0].film().set_crop_window() 

        self.scene.sensors()[0].film().set_crop_window((0,0), (render_dim, render_dim)) 

        renders = np.zeros((render_num, render_dim, render_dim, 3), dtype = "float32")
        

        for i in range(render_num):
            self.configure_pattern(pattern_param, patterns[i])

            self.render()

            renders[i] = self.write_to_numpy(sRGB = sRGB)

        white_ptrn = np.ones((512, 512, 3))
        self.configure_pattern(pattern_param, white_ptrn)
        self.render()

        white = self.write_to_numpy(sRGB = sRGB)
        
        
        black_ptrn = np.zeros((512, 512, 3))
        self.configure_pattern(pattern_param, black_ptrn)
        self.render()

        black = self.write_to_numpy(sRGB = sRGB)

        return renders, white, black

