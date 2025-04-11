import threading


class CaptureManager(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CaptureManager, self).__init__(name=name)
        self.stop_event = threading.Event()
        self.start()

    def stop(self):
        self.stop_event.set()

    def run(self):
        while not self.stop_event.is_set():
            _, self.last_frame = self.camera.read()
