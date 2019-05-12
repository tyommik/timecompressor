from collections import namedtuple
import sorting
import cv2
from itertools import combinations

Detection = namedtuple('Detection', 'pos, x, y, w, h, obj_class, precision')

video_input = r'timecompressor_archive.mp4'
video_output = r'summary_4.mp4'
cap = cv2.VideoCapture(video_input)
cap.set(2, 1500)

writeVideo_flag = True

class Event:
    def __init__(self, start):
        self.detections = []
        self.detections_map = {}
        self.__start = start
        self.__stop = None

    def __repr__(self):
        return "Event start={0} detection_length={1}".format(self.start, len(self.detections))

    def __hash__(self):
        return hash(self.start)

    @property
    def start(self):
        return self.__start

    @property
    def stop(self):
        return self.__stop or self.detections[-1].pos

    @property
    def length(self):
        return self.stop - self.start


class VideoCapture:
    def __init__(self, video_source, start_frame):
        self.video = cv2.VideoCapture(video_source)
        self.start_frame = start_frame

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
        f = self.get_position()
        hours = int(f // 90000)
        minutes = int((f - hours * 90000) // 1500)
        secondes = int((f - hours * 90000 - minutes * 1500) // 25)
        return "{0:02d}:{1:02d}:{2:02d}".format(hours, minutes, secondes)


def read_file():
    with open("detection.txt") as inf:
        for line in iter(inf):
            line = line.strip()
            pos, x, y, w, h, obj_class, precision = line.split(' ')
            if obj_class == '0':
                detection = Detection(int(pos), int(x), int(y), int(w), int(h), int(obj_class), float(precision))
                yield detection

def collect_events():
    events = []
    step = None
    for detection in read_file():
        if step is None:
            step = detection.pos
            event = Event(detection.pos)
            event.detections.append(detection)
            continue

        # если между детекциями меньше 5 секунд (20 * 25fps)- это детекции одной сцены
        if detection.pos - step <= 125:
            event.detections.append(detection)
        else:
            events.append(event)
            event = Event(detection.pos)
            event.detections.append(detection)

        step = detection.pos

    for event in events:
        for det in event.detections:
            event.detections_map.setdefault(det.pos, []).append(det)
    return events


def main(events):

    fps = 0.0

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(cap.get(3))
        h = int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(video_output, fourcc, 25, (w, h))


    while events:

        stack = []

        for _ in range(3):
            stack.append(events.pop(0))

        combs = combinations(stack, 2)

        stack = list(reversed(sorted(stack, key=lambda x: x.length)))
        a_b = sorting.search_optimal(events[0], events[1])
        a_c = sorting.search_optimal(events[0], events[2])

        shift = 3 * 25
        videos = [VideoCapture(video_source=video_input, start_frame=event.start - shift - pause) for event, pause in zip(stack,(0, a_b, a_c))]

        max_length = stack[0].length + 2 * shift + max(a_c, a_b)

        while max_length:
            frames = [video.get_frame() for video in videos]

            background = frames[0]

            for idx, event in enumerate(stack, start=0):
                detections = event.detections_map.get(videos[idx].get_position())
                if detections:
                    for detection in detections:
                        # cv2.rectangle(background, (int(detection.x), int(detection.y)),
                        #               (int(detection.x + detection.w), int(detection.y + detection.h)),
                        #               (255, 255, 255), 2)

                        x = detection.x
                        y = detection.y
                        w = detection.w
                        h = detection.h
                        background[y:y + h, x:x + w] = frames[idx][y:y + h, x:x + w]

                    for detection in detections:
                        cv2.putText(background, videos[idx].get_time(), (int(detection.x), int(detection.y)), 0,
                                    5e-3 * 200, (0, 255, 0), 2)


            # res_show = cv2.addWeighted(frames[0], 0.5, frames[1], 0.5, 2)
            # res_show = cv2.addWeighted(res_show, 0.5, frames[2], 0.5, 2)

            # frame = video.get_frame()
            cv2.imshow('test', background)

            max_length -= 1

            if writeVideo_flag:
                # save a frame
                out.write(background)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

    # writer.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    events = collect_events()
    main(events)
