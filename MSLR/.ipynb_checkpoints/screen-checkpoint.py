import numpy as np

class Screen_Rect:
    #rectangular screen 
    def __init__(self, center, x_axis_world, y_axis_world):
        self.center_world = center
        self.x_axis_world = x_axis_world
        self.y_axis_world = y_axis_world
    def get_screen_coord_from_image_coord(self, img_coord):
        #returns 3D world coordinate from a 2d image coordinate on the screen pattern
        #y axis (h) first then x axis (w)
        #img_coord [0,1]^2
                
        img_coord_x = np.tile(img_coord[:,:, 1:2], (1,1, 3)) #[Np, Np,3]
        img_coord_y = np.tile(img_coord[:,:, 0:1], (1,1,3)) #[Np, Np,3]
        
        x_axis =  np.tile(self.x_axis_world, (img_coord.shape[0],img_coord.shape[1], 1))  #[Np, Np, 3]
        y_axis = np.tile(self.y_axis_world, (img_coord.shape[0],img_coord.shape[1], 1)) #[Np, Np, 3]
        
        
        return self.center_world[np.newaxis,np.newaxis,:] - x_axis * (img_coord_x - 0.5) - y_axis * (img_coord_y - 0.5)
    
    def set_scene_param_from_screen(self, params):
        
        res = np.zeros((4,3), dtype = "float32")
        res[0] = self.center_world - 0.5 * self.x_axis_world + 0.5 * self.y_axis_world
        res[1] = self.center_world - 0.5 * self.x_axis_world - 0.5 * self.y_axis_world
        res[2] = self.center_world + 0.5 * self.x_axis_world - 0.5 * self.y_axis_world
        res[3] = self.center_world + 0.5 * self.x_axis_world + 0.5 * self.y_axis_world
        
        params["screen.vertex_positions_buf"] = res.ravel()
        params.update()
        

        