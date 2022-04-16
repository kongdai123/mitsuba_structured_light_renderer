import numpy as np

class sinusoid:
    def __init__(self, dim = 512, num_phases = 4, periods = np.array([1]), dtype ="float32"):
        self.dim = dim
        self.dtype = dtype 
        self.num_phases = num_phases
        self.periods = np.array([1])
        
    
    
    def generate_pattern(self):
        Nt = self.dim
        Nph = self.num_phases
        Nperiods = self.periods.shape[0]
        periods = np.array(self.periods)
        periods = np.transpose(np.tile(periods, (Nph,Nt,Nt,Nperiods)), axes = (3,0,1,2)).reshape(Nph*Nperiods, Nt, Nt) # Nph*Nperiods x Nt x Nt

        # the sampling grid for the sinusoids
        xp = np.linspace(-1,1,Nt) # grid is scaled -1,1
        Ym,Xm = np.meshgrid(xp, xp);

        # create images for cosines in x/y directions
        t = np.arange(0.,self.num_phases)/self.num_phases; # Nt - 1D coordinates in [0,1] range
        phase = 2*np.pi*np.transpose(np.tile(t, (Nperiods,Nt,Nt,1)), axes = (3,0,1,2)).reshape(Nph*Nperiods, Nt, Nt); # Nph*Nperiods x Nt x Nt
        Xarg = periods*np.pi*np.tile(Xm, (Nph*Nperiods, 1, 1)) + phase; # Nph*Nperiods x Nt x Nt
        Xcos = .5+.5*np.cos(Xarg); # Np*Nperiods x Nt x Nt
        Yarg = periods*np.pi*np.tile(Ym, (Nph*Nperiods, 1, 1))+ phase; # Nph*Nperiods x Nt x Nt
        Ycos = .5+.5*np.cos(Yarg); # Nph*Nperiods x Nt x Nt

        # concatenate X/Y texture maps and add 3 color channels
        patterns = np.concatenate((Xcos, Ycos), axis = 0) # 2*Nph*Nperiods x Nt x Nt
        # expand to create three color channels
        patterns = np.transpose(np.tile(patterns, (3,1,1,1)), axes = (1,2,3,0)) # (2*Nph*Nperiods, Nt, Nt, 3) 

        return patterns
    
    def decode(self, renders,  illuminance_mask, plane = False):
        
        pattern_coord = np.zeros((1024, 1024,2 ), dtype = self.dtype)
        renders_r = np.mean(renders, axis = 3)
        
        pattern_coord[:,:,0] = np.arctan2((renders_r[3] - renders_r[1]), (renders_r[0] - renders_r[2] + 1e-6))
        
        
        pattern_coord[:,:, 1] = np.arctan2((renders_r[7] - renders_r[5]), (renders_r[4] - renders_r[6]+ 1e-6))
        
        pattern_coord  =  pattern_coord / np.pi/2  + 0.5
                
        return pattern_coord

    
    
