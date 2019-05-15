import cv2


class VideoCapture:
    def __init__(self, video_source, start_frame, shift_time):
        self.video = cv2.VideoCapture(video_source)
        self.start_frame = start_frame
        self.shift = shift_time

        if not self.video.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = int(self.video.get(3))  # int
        self.height = int(self.video.get(4)) # int

        self.set_position(self.start_frame)
        self.ps_frame = None

    def set_position(self, position=0):
        if position:
            self.video.set(1, position)
        else:
            self.video.set(1, self.start_frame - 1)

    def get_position(self):
        return self.video.get(1)

    def get_frame(self):

        if self.video.isOpened():
            ret, frame = self.video.read()

            if ret:
                return frame
            else:
                print('No frame')

    def get_time(self):
        f = self.get_position() + self.shift
        hours = int(f // 90000)
        minutes = int((f - hours * 90000) // 1500)
        secondes = int((f - hours * 90000 - minutes * 1500) // 25)
        return "{0:02d}:{1:02d}:{2:02d}".format(hours, minutes, secondes)


def channel_generator(event_stack, video_input):

    while event_stack:
        event = event_stack.pop(0)
        video = VideoCapture(video_source=video_input, start_frame=event.start - 2 * 25)

        event_duration = event.stop - event.start

        while event_duration + 2 * 25:
            frame = video.get_frame()
            timestamp = video.get_time()
            event_duration -= 1
            det = event.detections_map.get(video.get_position(), [])
            yield (frame, det, timestamp, video.get_position())