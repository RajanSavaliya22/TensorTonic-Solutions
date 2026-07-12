import numpy as np
def morphological_op(image, kernel, operation):
    """
    Apply morphological erosion or dilation to a binary image.
    """
    image=  np.asarray(image, dtype=int)
    kernel=  np.asarray(kernel, dtype=int)
    padding_rows= kernel.shape[0] //2
    padding_columns= kernel.shape[1]//2

    padded = np.pad(
        image,
        ((padding_rows,padding_rows), (padding_columns,padding_columns)),
        mode='constant'
    )
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[
                i:i+kernel.shape[1],
                j:j+kernel.shape[0]
                ]

            values = window[kernel==1]
            if operation == 'erode':
                output[i,j] = 1 if np.all(values==1) else 0

            elif operation == 'dilate':
                output[i,j] = 1 if np.any(values==1) else 0

    return output.tolist()
                
        
        
            