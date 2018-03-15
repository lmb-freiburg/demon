import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport isfinite



@cython.boundscheck(False)
cdef _compute_visible_points_mask( 
        np.ndarray[np.float32_t, ndim=2] depth, 
        np.ndarray[np.float32_t, ndim=2] K1, 
        np.ndarray[np.float32_t, ndim=2] R1,
        np.ndarray[np.float32_t, ndim=1] t1,
        np.ndarray[np.float32_t, ndim=2] P2,
        int width2, 
        int height2,
        int borderx,
        int bordery):

    cdef np.float32_t point3d[3]
    cdef np.float32_t point4d[4]
    point4d[3] = 1.0
    cdef np.float32_t point_proj[3]
    cdef int x, y
    cdef np.float32_t px, py
    cdef np.float32_t d
    cdef np.ndarray[np.float32_t,ndim=2] RT = R1.transpose()

    cdef np.ndarray[np.uint8_t,ndim=2] mask = np.zeros((depth.shape[0],depth.shape[1]), dtype=np.uint8)

    for y in range(depth.shape[0]):
        for x in range(depth.shape[1]):

            d = depth[y,x]
            if np.isfinite(d) and d > 0.0:
                px = x + 0.5
                py = y + 0.5

                point3d[0] = d*(px - K1[0,2])/K1[0,0]
                point3d[1] = d*(py - K1[1,2])/K1[1,1]
                point3d[2] = d
                point3d[0] -= t1[0]
                point3d[1] -= t1[1]
                point3d[2] -= t1[2]
                point4d[0] = RT[0,0]*point3d[0] + RT[0,1]*point3d[1] + RT[0,2]*point3d[2]
                point4d[1] = RT[1,0]*point3d[0] + RT[1,1]*point3d[1] + RT[1,2]*point3d[2]
                point4d[2] = RT[2,0]*point3d[0] + RT[2,1]*point3d[1] + RT[2,2]*point3d[2]

                point_proj[0] = P2[0,0]*point4d[0] + P2[0,1]*point4d[1] + P2[0,2]*point4d[2] + P2[0,3]*point4d[3]
                point_proj[1] = P2[1,0]*point4d[0] + P2[1,1]*point4d[1] + P2[1,2]*point4d[2] + P2[1,3]*point4d[3]
                point_proj[2] = P2[2,0]*point4d[0] + P2[2,1]*point4d[1] + P2[2,2]*point4d[2] + P2[2,3]*point4d[3]
                if point_proj[2] > 0.0:
                    point_proj[0] /= point_proj[2]
                    point_proj[1] /= point_proj[2]
                    if point_proj[0] > borderx and point_proj[1] > bordery and point_proj[0] < width2-borderx and point_proj[1] < height2-bordery:
                        mask[y,x] = 1
                
    return mask



def compute_visible_points_mask( view1, view2, borderx=0, bordery=0 ):
    """Computes a mask of the pixels in view1 that are visible in view2

    view1: View namedtuple
        First view

    view2: View namedtuple
        Second view

    borderx: int
        border in x direction. Points in the border are considered invalid

    bordery: int
        border in y direction. Points in the border are considered invalid

    Returns a mask of valid points
    """
    assert view1.depth_metric == 'camera_z', "Depth metric must be 'camera_z'"
    
    P2 = np.empty((3,4), dtype=np.float32)
    P2[:,0:3] = view2.R
    P2[:,3:4] = view2.t.reshape((3,1))
    P2 = view2.K.dot(P2)

    if view2.depth is None:
        width2 = view1.depth.shape[1]
        height2 = view1.depth.shape[0]
    else:
        width2 = view2.depth.shape[1]
        height2 = view2.depth.shape[0]

    return _compute_visible_points_mask(
            view1.depth, 
            view1.K.astype(np.float32), 
            view1.R.astype(np.float32), 
            view1.t.astype(np.float32), 
            P2.astype(np.float32), 
            width2,
            height2,
            borderx,
            bordery)




@cython.boundscheck(False)
cdef _compute_depth_ratios( 
        np.ndarray[np.float32_t, ndim=2] depth1, 
        np.ndarray[np.float32_t, ndim=2] depth2, 
        np.ndarray[np.float32_t, ndim=2] K1, 
        np.ndarray[np.float32_t, ndim=2] R1,
        np.ndarray[np.float32_t, ndim=1] t1,
        np.ndarray[np.float32_t, ndim=2] P2 ):
    cdef np.float32_t point3d[3]
    cdef np.float32_t point4d[4]
    point4d[3] = 1.0
    cdef np.float32_t point_proj[3]
    cdef int x, y, x2, y2
    cdef np.float32_t px, py
    cdef np.float32_t d, d2
    cdef np.ndarray[np.float32_t,ndim=2] RT = R1.transpose()

    cdef np.ndarray[np.float32_t,ndim=2] result = np.full((depth1.shape[0],depth1.shape[1]), np.nan, dtype=np.float32)

    for y in range(depth1.shape[0]):
        for x in range(depth1.shape[1]):

            d = depth1[y,x]
            if np.isfinite(d) and d > 0.0:
                px = x + 0.5
                py = y + 0.5

                point3d[0] = d*(px - K1[0,2])/K1[0,0]
                point3d[1] = d*(py - K1[1,2])/K1[1,1]
                point3d[2] = d
                point3d[0] -= t1[0]
                point3d[1] -= t1[1]
                point3d[2] -= t1[2]
                point4d[0] = RT[0,0]*point3d[0] + RT[0,1]*point3d[1] + RT[0,2]*point3d[2]
                point4d[1] = RT[1,0]*point3d[0] + RT[1,1]*point3d[1] + RT[1,2]*point3d[2]
                point4d[2] = RT[2,0]*point3d[0] + RT[2,1]*point3d[1] + RT[2,2]*point3d[2]

                point_proj[0] = P2[0,0]*point4d[0] + P2[0,1]*point4d[1] + P2[0,2]*point4d[2] + P2[0,3]*point4d[3]
                point_proj[1] = P2[1,0]*point4d[0] + P2[1,1]*point4d[1] + P2[1,2]*point4d[2] + P2[1,3]*point4d[3]
                point_proj[2] = P2[2,0]*point4d[0] + P2[2,1]*point4d[1] + P2[2,2]*point4d[2] + P2[2,3]*point4d[3]
                if point_proj[2] > 0.0:
                    point_proj[0] /= point_proj[2]
                    point_proj[1] /= point_proj[2]
                    if point_proj[0] > 0 and point_proj[1] > 0 and point_proj[0] < depth2.shape[1] and point_proj[1] < depth2.shape[0]:
                        # lookup the depth value
                        x2 = max(0,min(depth2.shape[1],int(round(point_proj[0]))))
                        y2 = max(0,min(depth2.shape[0],int(round(point_proj[1]))))
                        d2 = depth2[y2,x2]
                        if d2 > 0.0 and isfinite(d2):
                            s = point_proj[2]/d2
                            result[y,x] = s
                
    return result
    



def compute_depth_ratios( view1, view2 ):
    """Projects each point defined in view1 to view2 and computes the ratio of 
    the depth value of the projected point and the stored depth value in view2.


    view1: View namedtuple
        First view

    view2: View namedtuple
        Second view

    Returns the scale value for view2 relative to view1
    """
    assert view1.depth_metric == 'camera_z', "Depth metric must be 'camera_z'"
    assert view2.depth_metric == 'camera_z', "Depth metric must be 'camera_z'"
    
    P2 = np.empty((3,4), dtype=np.float32)
    P2[:,0:3] = view2.R
    P2[:,3:4] = view2.t.reshape((3,1))
    P2 = view2.K.dot(P2)

    return _compute_depth_ratios(
            view1.depth, 
            view2.depth,
            view1.K.astype(np.float32), 
            view1.R.astype(np.float32), 
            view1.t.astype(np.float32), 
            P2.astype(np.float32) )



@cython.boundscheck(False)
cdef _compute_flow( 
        np.ndarray[np.float32_t, ndim=2] depth1, 
        np.ndarray[np.float32_t, ndim=2] K1, 
        np.ndarray[np.float32_t, ndim=2] R1,
        np.ndarray[np.float32_t, ndim=1] t1,
        np.ndarray[np.float32_t, ndim=2] P2 ):
    cdef np.float32_t point3d[3]
    cdef np.float32_t point4d[4]
    point4d[3] = 1.0
    cdef np.float32_t point_proj[3]
    cdef int x, y, x2, y2
    cdef np.float32_t px, py
    cdef np.float32_t d, d2
    cdef np.ndarray[np.float32_t,ndim=2] RT = R1.transpose()

    cdef np.ndarray[np.float32_t,ndim=3] result = np.full((2,depth1.shape[0],depth1.shape[1]), np.nan, dtype=np.float32)

    for y in range(depth1.shape[0]):
        for x in range(depth1.shape[1]):

            d = depth1[y,x]
            if np.isfinite(d) and d > 0.0:
                px = x + 0.5
                py = y + 0.5

                point3d[0] = d*(px - K1[0,2])/K1[0,0]
                point3d[1] = d*(py - K1[1,2])/K1[1,1]
                point3d[2] = d
                point3d[0] -= t1[0]
                point3d[1] -= t1[1]
                point3d[2] -= t1[2]
                point4d[0] = RT[0,0]*point3d[0] + RT[0,1]*point3d[1] + RT[0,2]*point3d[2]
                point4d[1] = RT[1,0]*point3d[0] + RT[1,1]*point3d[1] + RT[1,2]*point3d[2]
                point4d[2] = RT[2,0]*point3d[0] + RT[2,1]*point3d[1] + RT[2,2]*point3d[2]

                point_proj[0] = P2[0,0]*point4d[0] + P2[0,1]*point4d[1] + P2[0,2]*point4d[2] + P2[0,3]*point4d[3]
                point_proj[1] = P2[1,0]*point4d[0] + P2[1,1]*point4d[1] + P2[1,2]*point4d[2] + P2[1,3]*point4d[3]
                point_proj[2] = P2[2,0]*point4d[0] + P2[2,1]*point4d[1] + P2[2,2]*point4d[2] + P2[2,3]*point4d[3]

                point_proj[0] /= point_proj[2]
                point_proj[1] /= point_proj[2]
                result[0,y,x] = point_proj[0]-px
                result[1,y,x] = point_proj[1]-py
                
    return result
    
