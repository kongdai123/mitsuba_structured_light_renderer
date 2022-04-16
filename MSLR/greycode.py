import numpy as np


def binary_to_grey(n): 
 
    return n ^ (n >> 1) 

def gray_to_binary(n): 
    if n < 0:
        return n
    mask = n
    while mask != 0:
        mask >>= 1
        n ^= mask
 
    return n


def img_rgb_distance(img, color):
    color_diff = (img - color[np.newaxis, np.newaxis, :]) 
    return np.sum(color_diff * color_diff, axis = 2)

class greyCode:
    def __init__(self, dim, num_patterns, colors, axis = "X", dtype ="float32"):
        self.dim = dim
        self.axis = axis
        self.dtype = dtype 
        self.num_patterns = num_patterns
        self.binaries = 1 << self.num_patterns
        self.colors = colors
        
    
    
    def generate_pattern(self):
        colors = self.colors
        binaries = 1 << self.num_patterns
        grey_codes = np.array([binary_to_grey(i) for i in range(binaries)])
        patterns = np.zeros((self.num_patterns, self.dim, self.dim, 3), dtype = self.dtype)
        
        for i in range(self.num_patterns):
            mask = i 
            bit = (grey_codes >> mask) % 2

            bit = np.transpose(np.tile(bit, (3,1)))

            bit = colors[1][np.newaxis,: ] * bit + colors[0][np.newaxis,: ] * (1 - bit)
            
            bit = np.tile(bit, (binaries, 1, 1))

            
            if self.axis == "Y":
                bit = bit.transpose(1,0,2)
    
                
            bit = np.repeat(bit, self.dim/binaries, axis = 0)
            bit = np.repeat(bit, self.dim/binaries, axis = 1)
            
            patterns[i][0:bit.shape[0],0:bit.shape[1],: ] = bit
        
        
        
        
        return patterns

    
    def decode(self, renders,  illuminance_mask, plane = False):
        stripe_num = np.zeros_like(renders[0][:,:,0])
        base = 1
        for i in range(renders.shape[0]):
            stripe_num += illuminance_mask * base * np.where( renders[i,:,:, 2] >= renders[i,:,:, 1], np.ones_like(renders[0][:,:,0]), np.zeros_like(renders[0][:,:,0]))
            base = base * 2

        #make sure areas not covered are negative
        stripe_num  = stripe_num - 1000 * (1 - illuminance_mask)
        
        stripe_num = stripe_num.astype(int)
        vfunc = np.vectorize(gray_to_binary)
        stripe_num = vfunc(stripe_num)

        pattern_coord =  stripe_num * (1/self.binaries )
        
        if plane:
            pc = pattern_coord.reshape((pattern_coord.shape[0], pattern_coord.shape[1], 1))
            if self.axis == 'X':
                point_1 = np.concatenate((np.zeros_like(pc), pc), axis = 2)

                point_2 = np.concatenate((np.ones_like(pc), pc), axis = 2)

            if self.axis == 'Y':
                point_1 = np.concatenate((pc, (np.zeros_like(pc))), axis = 2)

                point_2 = np.concatenate((pc, (np.ones_like(pc))), axis = 2)
            return np.concatenate((point_1[np.newaxis, :,:,:], point_2[np.newaxis, :,:,:]), axis = 0)
                
        return pattern_coord

        
        
        
        
        
        
