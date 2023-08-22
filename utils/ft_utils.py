import numpy as np
import time
from recording.ft import FTCapture

def calibrate_ft(ft_obj):
    ft = ft_obj.get_ft()
    np.save('ft_calibration.npy', ft)

def get_ft_calibration():
    return np.load('ft_calibration.npy')

if __name__ == '__main__':
    robot = None
    ft_obj = FTCapture()
    calibrate_ft(ft_obj)
    offset = get_ft_calibration()

    while True:
        ft = ft_obj.get_ft()
        ft = ft - offset
        print('calibrated FT: ', ft)