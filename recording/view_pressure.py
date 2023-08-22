import os
import pickle
import cv2
import numpy as np
from utils.config_utils import *
from utils.camera_utils import *
from utils.sensel_utils import *
from utils.recording_utils import *
from recording.aruco import ArucoGridEstimator
import recording.sensel_wrapper as sensel_wrapper
import pickle


def main():
    config, args = parse_config_args()
    pose_estimator = ArucoGridEstimator(config)
    sensel_obj = sensel_wrapper.SenselWrapper()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ret, img = cap.read()

    if args.record_video:
        video_index = len(os.listdir('videos')) + 1
        result = cv2.VideoWriter('video_{}.avi'.format(video_index), cv2.VideoWriter_fourcc(*'MJPG'), 20, (img.shape[1], img.shape[0]))

    while True:
        ret, img = cap.read()
        if img is not None:
            image, homography, rvec, tvec, imgpts, ids = pose_estimator.get_cam_params_aruco(img)
        
        mtx, dist = get_camera_calibration(config)
        img = cv2.undistort(img, mtx, dist, None, mtx)

        if homography is not None:
            raw_force = sensel_obj.cur_force_array
            print('raw_force: ', raw_force.dtype)
            print('raw_force shape: ', raw_force.shape)
            pressure_kPa = convert_counts_to_kPa(raw_force)
            newtons = convert_kPa_to_newtons(pressure_kPa)

            # print(f'Pixels on: {(pressure_kPa > 0).sum()}, sum force newton {newtons.sum()}, max pressure kPa {pressure_kPa.max()}, max raw {raw_force.max()}, max newton {newtons.max()}')

            pressure_kPa[pressure_kPa < 0] = 0

            # rgb = cv2.resize(pressure_to_colormap(pressure_kPa), (disp_x, disp_y), cv2.INTER_NEAREST) / 255.0
            # cv2.imshow("Cage recording", rgb)

            overlay = get_force_overlay_img(img, pressure_kPa, homography)
            cv2.imshow("Pressure", overlay)

            if args.record_video:
                result.write(overlay)

            keycode = cv2.waitKey(10) & 0xFF
            if keycode == ord('q'):
                break
        else:
            cv2.imshow("Pressure", img)
            if args.record_video:
                result.write(img)

            keycode = cv2.waitKey(10) & 0xFF
            if keycode == ord('q'):
                break

    cv2.destroyAllWindows()
    sensel_obj.close_sensel()  
    if args.record_video:
        result.release()

if __name__ == '__main__':
    main()