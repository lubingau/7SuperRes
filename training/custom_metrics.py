from tensorflow.image import psnr

def PSNR(y_true, y_pred):
    return psnr(y_true, y_pred, max_val=1.0)