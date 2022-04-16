import numpy as np

def getRotationMatrix(matrix):
    #Matrix is a 4X4 transform matrix
    rot = np.zeros((3,3), dtype="float32")
    for i in range(3):
        for j in range(3):
            rot[i,j] = matrix[i, j]
    return rot

def getNpProjMat(matrix):
    string = str(matrix)
    string = string.replace('[', '').replace(']', '')
    rot = np.fromstring(string, dtype="float32", sep=', ').reshape(4,4)
    return rot


def generate_camera_coords(img_dim):
    y_coord = np.linspace(0,img_dim[1] - 1, img_dim[1]) /img_dim[1]
    x_coord = np.linspace(0,img_dim[0] - 1, img_dim[0]) /img_dim[0]
    xv, yv = np.meshgrid(x_coord,y_coord)
    return  np.concatenate((yv[:, :, np.newaxis], xv[:, :, np.newaxis]), axis = 2)


class Camera:
    ################################################################
    # persepctive camera object
    #
    # inputs:
    # proj_to_world - 4X4 camera space to world space projection matrix
    # fov_x- field of view in x direction in degrees
    # fov_y - field of view in ydirection in degrees
    #
    ################################################################
    def __init__(self, proj_to_world, fov_x, fov_y):
        self.view_to_world = proj_to_world
        self.world_to_view = np.linalg.inv(proj_to_world) 
        self.x_axis_world = proj_to_world[0:3,0]
        self.y_axis_world = proj_to_world[0:3,1]
        self.z_axis_world = proj_to_world[0:3,2]
        self.origin = proj_to_world[0:3, 3]
        
        #fov should be in RADIANS
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.dtype = 'float32'
    
    def get_world_space_coord_from_img_coord(self, img_coord_x, img_coord_y, z_dist):
        #img coord is from [0,1]^2
        #x is the width axis, goes from left to right
        #y is the hight axis, goes from up to down
        center = self.origin + self.z_axis_world * z_dist
        x_dist = z_dist * np.tan(self.fov_x/2) * 2
        y_dist = z_dist * np.tan(self.fov_y/2) * 2
        return center - self.x_axis_world * x_dist * (img_coord_x - 0.5) - self.y_axis_world * y_dist * (img_coord_y - 0.5)
    
    def get_world_space_coord_from_img_coord_vectorized(self, img_coord, z_dist = 1):
        #img coord is from [0,1]^2
        #x is the width axis, goes from left to right
        #y is the hight axis, goes from up to down
        #img_coord is [Np, Np, 2], img_coord_y, img_coord_x
        center = self.origin + self.z_axis_world * z_dist
        
        img_coord_x = np.tile(img_coord[:,:, 1:2], (1,1, 3)) #[Np, Np,3]
        img_coord_y = np.tile(img_coord[:,:, 0:1], (1,1,3)) #[Np, Np,3]
        
        x_dist = z_dist * np.tan(self.fov_x/2) * 2
        y_dist = z_dist * np.tan(self.fov_y/2) * 2
        
        x_axis =  np.tile(self.x_axis_world, (img_coord.shape[0],img_coord.shape[1], 1)) * x_dist #[Np, Np, 3]
        y_axis = np.tile(self.y_axis_world, (img_coord.shape[0],img_coord.shape[1], 1))  * y_dist #[Np, Np, 3]
        
        
        return center[np.newaxis,np.newaxis,:] - x_axis * (img_coord_x - 0.5) - y_axis * (img_coord_y - 0.5)
    
    
    def get_img_coord_from_world_coord(self, world_coord, img_dim):
        #in mitsuba camera corrdinates, x goes from right to left 
        #y goes from down to up,and z goes in the same direction as viewing direction
        cam_coord = np.dot(self.world_to_view, world_coord)
    
        cam_coord[1] /= cam_coord[2]
        cam_coord[0] /= cam_coord[2]
        
        imsize_x = img_dim[1]
        imsize_y = img_dim[0]
        
        

        #computing the focal length in pixels
        flength_x = 1 /(np.tan(self.fov_x * 0.5) * 2 /imsize_x)
        flength_y = 1 /(np.tan(self.fov_y * 0.5) * 2 /imsize_y)
        
        
        x_coord =  1 - ((flength_x * cam_coord[0] ) /imsize_x + 0.5)
        y_coord =   1 - ((flength_y * cam_coord[1] ) /imsize_y + 0.5)
        
        return x_coord, y_coord
        
        
        
        
        
        
        
    

        