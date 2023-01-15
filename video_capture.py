import threading


class CaptureManager(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CaptureManager, self).__init__(name=name)
        self.start()
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.isSet():
            ret, self.last_frame = self.camera.read()
