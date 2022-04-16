import matplotlib.pyplot as plt
import numpy as np


def compare_depth_map(depth_cal, depth_gt, channel = 0):
    coordinates = ["x", "y", "z"]
    fig, axs = plt.subplots(1,2,figsize=(20,10))
    im = axs[1].imshow(depth_gt[:,:,channel])
    axs[1].set_title('ground truth world space location, ' + coordinates[channel] + ' coordinate') 
    clim=im.properties()['clim']
    print(clim)
    axs[0].set_title('structured light world space location, ' + coordinates[channel] + ' coordinate') 
    axs[0].imshow(depth_cal[:,:,channel], clim = clim)
    
    
    fig.colorbar(im, ax=axs.ravel().tolist())
    plt.show()