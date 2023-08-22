import cv2
import numpy as np
import recording.sensel_wrapper as sensel_wrapper
import pickle
from datetime import datetime
import argparse
import os
from pathlib import Path
import threading
import time
from collections import OrderedDict
from utils.sensel_utils import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


disp_x = 1280    # Rendering X, y
disp_y = 720


thresh = 1.0

def main():
    sensel_obj = sensel_wrapper.SenselWrapper()

    while True:
        raw_force = sensel_obj.cur_force_array
        pressure_kPa = convert_counts_to_kPa(raw_force)
        newtons = convert_kPa_to_newtons(pressure_kPa)

        print(f'Pixels on: {(pressure_kPa > 0).sum()}, sum force newton {newtons.sum()}, max pressure kPa {pressure_kPa.max()}, max raw {raw_force.max()}, max newton {newtons.max()}')

        pressure_kPa[pressure_kPa < 0] = 0

        rgb = cv2.resize(pressure_to_colormap(pressure_kPa), (disp_x, disp_y), cv2.INTER_NEAREST) / 255.0
        cv2.imshow("Cage recording", rgb)

        keycode = cv2.waitKey(10) & 0xFF
        if keycode == ord('q'):
            break

    cv2.destroyAllWindows()
    sensel_obj.close_sensel()


if __name__ == "__main__":
    main()