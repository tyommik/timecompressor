from collections import namedtuple
import time
import numpy as np
import cv2

Detection = namedtuple('Detection', 'pos, x, y, w, h, obj_class, precision')

video_input = r'timecompressor_archive.mp4'
video_output = r'summary.mp4'
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
        frame_index = -1

    video = VideoCapture(video_source=video_input, start_frame=0)
    start_pos = 0

    for idx, event in enumerate(events):

        video.set_position(start_pos)

        gap_step = (event.start - start_pos) // (1 * 25)

        total = 30
        (rAvg, gAvg, bAvg, mAvg) = (None, None, None, None)
        # проматываем на ускорении гап между событиями
        # накладываем кадры друг на друга, чтобы было плавно
        while start_pos < event.start - 3 * 25:
            print(video.get_position())

            frame = video.get_frame()
            (B, G, R) = cv2.split(frame.astype("float"))
            # if the frame averages are None, initialize them
            if rAvg is None:
                rAvg = R
                bAvg = B
                gAvg = G

            # otherwise, compute the weighted average between the history of
            # frames and the current frames
            else:
                rAvg = ((total * rAvg) + (1 * R)) / (total + 1.0)
                gAvg = ((total * gAvg) + (1 * G)) / (total + 1.0)
                bAvg = ((total * bAvg) + (1 * B)) / (total + 1.0)

            # increment the total number of frames read thus far
            # total += 1

            avg = cv2.merge([bAvg, gAvg, rAvg]).astype("uint8")

            start_pos += gap_step
            video.set_position(start_pos)
            cv2.imshow('heatmap', avg)

            if writeVideo_flag:
                # save a frame
                out.write(frame)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

        while video.get_position() <= event.stop + 3 * 25:
            t1 = time.time()

            frame = video.get_frame()

            detections = event.detections_map.get(video.get_position())
            if detections:
                for detection in detections:
                    print('Video frame: ', video.get_position(), 'event_frame: ', detection.pos)
                    cv2.rectangle(frame, (int(detection.x), int(detection.y)), (int(detection.x + detection.w), int(detection.y + detection.h)),
                                  (255, 255, 255), 2)
                    # cv2.putText(frame, str(detection.obj_class_name), (int(detection.x), int(detection.y)), 0, 5e-3 * 200, (0, 255, 0), 2)

            cv2.imshow('heatmap', frame)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %f" % (fps))

            if writeVideo_flag:
                # save a frame
                out.write(frame)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        start_pos = video.get_position() + 1

    # writer.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    events = collect_events()
    main(events)
