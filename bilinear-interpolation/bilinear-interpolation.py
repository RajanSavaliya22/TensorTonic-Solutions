import numpy as np
def bilinear_resize(image, new_h, new_w):
    """
    Resize a 2D grid using bilinear interpolation.
    """
    image= np.asarray(image)
    output = np.zeros((new_h,new_w))

    H, W = image.shape
    for i in range(new_h):
        for j in range(new_w):
            src_x = j*(W-1)/(new_w - 1) if new_w >1 else 0
            src_y= i*(H - 1)/(new_h - 1) if new_h > 1 else 0
            
            x0 = int(src_x)
            y0 = int(src_y)
            x1 = min(x0 + 1, W-1)
            y1 = min(y0 + 1, H-1)
            
            dx = src_x - x0
            dy = src_y - y0

            output[i,j] = image[y0,x0]*(1-dy)*(1-dx) + image[y1,x0]*dy*(1-dx) + image[y0,x1]*(1-dy)*dx + image[y1,x1]*dy*dx
    return output.tolist()