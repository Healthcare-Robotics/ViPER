import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import threading
import recording.sensel as sensel

# try:
#     import recording.sensel as sensel
# except:
#     pass

class SenselWrapper:
    sensel_width = 185
    sensel_height = 105

    def __init__(self):
        self.handle = None
        (error, device_list) = sensel.getDeviceList()
        if device_list.num_devices != 0:
            (error, self.handle) = sensel.openDeviceByID(device_list.devices[0].idx)

        if self.handle is None:
            raise IOError('Could not connect to sensel')

        (error, self.info) = sensel.getSensorInfo(self.handle)

        error = sensel.setFrameContent(self.handle, sensel.FRAME_CONTENT_PRESSURE_MASK)
        (error, self.frame) = sensel.allocateFrameData(self.handle)
        error = sensel.startScanning(self.handle)

        self.cur_force_array = np.zeros((self.sensel_height, self.sensel_width))
        self.cur_force_timestamp = 0

        self.thread = DaemonStoppableThread(0.005, target=self.scan_frames, name='polling_thread')
        self.thread.start()

    def scan_frames(self):
        # This function takes 10-20 ms
        t1 = time.time()
        error = sensel.readSensor(self.handle)
        (error, num_frames) = sensel.getNumAvailableFrames(self.handle)

        if num_frames == 0:
            return None

        for i in range(num_frames):
            self.cur_force_timestamp = time.time()
            error = sensel.getFrame(self.handle, self.frame)
            force_array = np.zeros((self.info.num_rows, self.info.num_cols))

            for y in range(self.info.num_rows):
                for x in range(self.info.num_cols):
                    force_array[y, x] = self.frame.force_array[x + y * self.info.num_cols]

            self.cur_force_array = force_array

        # print('sensel reading time', time.time() - t1)

        # print('max, min, mean', force_array.max(), force_array.min(), force_array.mean())
        # print(num_frames, time.time() - start_time, time.time() - t1)
        # return force_array
        # ind = np.unravel_index(np.argmax(force_array, axis=None), force_array.shape)
        # print('max force', ind)

    def close_sensel(self):
        self.thread.stop()

        error = sensel.freeFrameData(self.handle, self.frame)
        error = sensel.stopScanning(self.handle)
        error = sensel.close(self.handle)

class DaemonStoppableThread(threading.Thread):
    def __init__(self, sleep_time, target=None,  **kwargs):
        super(DaemonStoppableThread, self).__init__(target=target, **kwargs)
        self.setDaemon(True)
        self.stop_event = threading.Event()
        self.sleep_time = sleep_time
        self.target = target

    def stop(self):
        self.stop_event.set()

    def stopped(self):
        return self.stop_event.isSet()

    def run(self):
        while not self.stopped():
            if self.target:
                self.target()
            else:
                raise Exception('No target function given')
            self.stop_event.wait(self.sleep_time)