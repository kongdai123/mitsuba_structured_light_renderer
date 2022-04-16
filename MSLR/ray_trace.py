import numpy as np


def plane_normal_from_3_pts(p1,p2,p3):
    #specify a plane normal from 3 points on the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)

    
    return cp/np.sqrt(np.dot(cp,cp))

def plane_normal_from_3_pts_vectorized(p1,p2,p3):
    #specify a plane normal from 3 points on the plane
    #p1, p2, p3 are [Np, Np, 3] arrays
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)

    
    return cp/np.linalg.norm(cp, axis = 2)[:, :, np.newaxis]


def LinePlaneCollision(planeNormal, planePoint, rayDir, rayPoint, epsilon=1e-6):
    
    #calculates the intersection of a plane and a line
    
    ndotu = planeNormal.dot(rayDir)
#     if abs(ndotu) < epsilon:
#         raise RuntimeError("no intersection or line is within plane")
 
    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDir + planePoint
    return Psi



def LinePlaneCollision_vectorized(planeNormal, planePoint, rayDir, rayPoint, epsilon=1e-6):
    
    #calculates the intersection of a plane and a line
    ndotu = np.sum(planeNormal * rayDir, axis = 2) 
 
    w = rayPoint - planePoint
    
    si = -np.sum(planeNormal * w, axis = 2) / ndotu
    
    Psi = w + si[:,:, np.newaxis] * rayDir + planePoint
    
    return Psi


def structured_light_ray_trace_vectorized(projector, proj_coord, camera, camera_coord):

    p1 = projector.get_world_space_coord_from_img_coord_vectorized(proj_coord[0], 1)
        
    p2 = projector.get_world_space_coord_from_img_coord_vectorized(proj_coord[1], 1)
        
    plane_normal = (plane_normal_from_3_pts_vectorized(p1,p2,projector.origin[np.newaxis,np.newaxis,:]))

    ray_dir = camera.get_world_space_coord_from_img_coord_vectorized(camera_coord , 1) -  camera.origin[np.newaxis,np.newaxis,:]
    
    ray_dir = ray_dir/np.linalg.norm(ray_dir, axis = 2)[:, :, np.newaxis]

    return LinePlaneCollision_vectorized(plane_normal, projector.origin[np.newaxis,np.newaxis,:], ray_dir, camera.origin[np.newaxis,np.newaxis,:])



def def_ray_trace_vectorized(points, camera, camera_coord):
    

    p1 = points[0]
        
    p2 = points[1]
    
    p3 = points[2]
        
    plane_normal = (plane_normal_from_3_pts_vectorized(p1,p2,p3))

    ray_dir = camera.get_world_space_coord_from_img_coord_vectorized(camera_coord , 1) -  camera.origin[np.newaxis,np.newaxis,:]
    

    return LinePlaneCollision_vectorized(plane_normal, p1, ray_dir, camera.origin[np.newaxis,np.newaxis,:])

def intersect(P0,P1):
    ###
    #P0 and P1 are NxD arrays defining N lines.
    #D is the dimension of the space. This function 
    #returns the least squares intersection of the N
    #lines from the system given by eq. 13 in 
    # http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf.
    ###
    # generate all line direction vectors 
    n = (P1-P0)/np.linalg.norm(P1-P0,axis=1)[:,np.newaxis] # normalized

    # generate the array of all projectors 
    projs = np.eye(n.shape[1]) - n[:,:,np.newaxis]*n[:,np.newaxis]  # I - n*n.T
    # see fig. 1 

    # generate R matrix and q vector
    R = projs.sum(axis=0)
    q = (projs @ P0[:,:,np.newaxis]).sum(axis=0)

    # solve the least squares problem for the 
    # intersection point p: Rp = q
    p = np.linalg.lstsq(R,q,rcond=None)[0]

    return p