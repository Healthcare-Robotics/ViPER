import os
import cv2
from pathlib import Path

def mkdir(path, cut_filename=False):
    if cut_filename:
        path = os.path.dirname(os.path.abspath(path))
    Path(path).mkdir(parents=True, exist_ok=True)

class MovieWriter:
    def __init__(self, path, fps=30):
        self.writer = None
        self.path = path
        self.fps = fps

    def write_frame(self, frame):
        if self.writer is None:
            mkdir(self.path, cut_filename=True)
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps, (frame.shape[1], frame.shape[0]))
        self.writer.write(frame)

    def close(self):
        self.writer.release()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)