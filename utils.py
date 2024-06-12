import numpy as np

def get_ray(M, R, t, p):
    """
    Get the ray vector from the camera center to the image point p
    """
    Rt = np.array(
        [
            [R[0, 0], R[0, 1], R[0, 2], t[0, 0]],
            [R[1, 0], R[1, 1], R[1, 2], t[1, 0]],
            [R[2, 0], R[2, 1], R[2, 2], t[2, 0]],
            [0, 0, 0, 1]
        ]
    )
    
    if p.shape != (3, 1):
       p = p.reshape(-1, 1)
       
    if p.shape == (2, 1):
        p = np.concatenate((p, [[1]]))
    
    Rt_inv = np.linalg.inv(Rt)
    
    cam_center = np.dot(Rt_inv, np.array([0, 0, 0, 1]).reshape(-1, 1))
    p = np.dot(np.linalg.inv(M), p)
    cam_pointer = np.dot(Rt_inv, np.concatenate((p, [[1]])))
    ray_vector = cam_pointer - cam_center
    ray_vector /= np.linalg.norm(ray_vector)
    return ray_vector[:3], cam_center[:3]


def get_least_square_point(pts, vecs):
    """
    Get the least square point from the points and directions
    """
    assert pts.shape == vecs.shape
    assert pts.shape[1] == 3
    if pts.shape[0] < 2:
        return None

    n = pts.shape[0]
    vecs = vecs / np.sqrt((vecs ** 2).sum(axis=-1, keepdims=True))
    
    da = (vecs * pts).sum(axis=-1, keepdims=True)
    b = (vecs * da - pts).sum(axis=0)
    c, d = pts.shape
    m = np.inner(vecs.T, vecs.T) - np.diag(np.full(d, c))
    return np.linalg.solve(m, b).reshape(-1, 1)