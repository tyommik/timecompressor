from collections import namedtuple
import sorting
import cv2
import numpy as np
from itertools import combinations
from enum import Enum, auto

CONST_TIME = 539975
Detection = namedtuple('Detection', 'pos, x, y, w, h, obj_class, precision')

video_input = r'/home/ashibaev/Downloads/timecompressor_archive.mp4'
video_output = r'summary_7.mp4'
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
        f = self.get_position() + CONST_TIME
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


def channel_generator(event_stack):

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



class State(Enum):
    WAITING = auto()
    RUNNING = auto()

class Op(Enum):
    STOP = auto()

class VideoChannel:

    def __init__(self, stack):
        self._stack = stack
        self._state = State.WAITING

    def run(self):
        self._state = State.RUNNING

    def __iter__(self):
        return self

    def step(self):
        while self._stack:
            event = self._stack.pop(0)

            # video = VideoCapture(video_source=video_input, start_frame=event.start - 2 * 25)
            video = VideoCapture(video_source=video_input, start_frame=event.start)

            event_duration = event.stop - event.start

            while event_duration:
                frame = video.get_frame()
                timestamp = video.get_time()
                event_duration -= 1
                det = event.detections_map.get(video.get_position(), [])

                # шлём один и тот же кадр пока нам не поменяют состояние

                if det:
                    while self._state is State.WAITING:
                        yield (frame, det, timestamp, video.get_position())

                yield (frame, det, timestamp, video.get_position())

            # событие кончилось, возвращаемся на исходную
            self._state = State.WAITING

            # предупреждаем, что у нас произошел сброс
            yield Op.STOP


def main(events):

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(cap.get(3))
        h = int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(video_output, fourcc, 25, (w, h))

    channels_number = 3

    while events:
        flag = True

        background = None

        working_channels = []
        waiting_channels = [VideoChannel(events) for _ in range(channels_number)]

        working_channels.append(waiting_channels.pop(0))

        for w in working_channels:
            w.run()

        working_channels_gens = [ch.step() for ch in working_channels]
        waiting_channels_gens = [ch.step() for ch in waiting_channels]

        while flag:

            if not flag:
                break

            for w in working_channels:
                w.run()

            image_stack = []

            for gen_idx, gen in enumerate([ch for ch in working_channels_gens]):
                # frame, det, timestamp = next(gen)

                tp = next(gen)
                if tp is Op.STOP:
                    working_channels.pop(gen_idx)
                    working_channels_gens.pop(gen_idx)

                    print('Событие закончилось')

                    new_ch = VideoChannel(events)
                    waiting_channels.append(new_ch)
                    waiting_channels_gens.append(new_ch.step())

                    continue
                if tp is not None:
                    image_stack.append(tp)
                else:
                    image_stack = []

            if image_stack:
                image_stack.sort(key=lambda x: x[3] if x is not None else 0)
            else:
                break

            if image_stack:

                pass_or_not = False
                for gen_idx, (frame, det, timestamp, pos) in enumerate(image_stack):
                    if gen_idx == 0:
                        background = frame
                    if frame is not None:
                        if det:
                            for detection in det:

                                if waiting_channels:
                                    check_channel, check_channel_gen = waiting_channels.pop(0), waiting_channels_gens.pop(0)

                                    _, w_dets, *args = next(check_channel_gen)

                                    for w_det in w_dets:
                                        res = sorting.check_rectangels(w_det, detection)
                                        if res <= 20:
                                            print('Площадь пересчения равна нулю')
                                            pass_or_not = True# pass_or_not * True
                                        else:
                                            print('Площадь пересечения равна ', res)

                                    if pass_or_not:

                                        print('Вставляем событие')

                                        working_channels.append(check_channel)
                                        working_channels_gens.append(check_channel_gen)
                                        break
                                    else:
                                        waiting_channels.insert(0, check_channel)
                                        waiting_channels_gens.insert(0, check_channel_gen)

                                        # waiting_channels.append(check_channel)
                                        # waiting_channels_gens.append(check_channel_gen)



                                # cv2.rectangle(frame, (int(detection.x), int(detection.y)),
                                #               (int(detection.x + detection.w), int(detection.y + detection.h)),
                                #               (255, 255, 255), 2)

                                x = detection.x
                                y = detection.y
                                w = detection.w
                                h = detection.h

                                multiplier = 0.1
                                
                                cv2.addWeighted(frame[y - int(multiplier * h): y + h + int(multiplier * h), x - int(multiplier * w): x + w + int(multiplier * w)], 0.7, background[y - int(multiplier * h): y + h + int(multiplier * h), x - int(multiplier * w): x + w + int(multiplier * w)], 0.3, 0, background[y - int(multiplier * h): y + h + int(multiplier * h), x - int(multiplier * w): x + w + int(multiplier * w)])


                            for detection in det:
                                multiplier = 0.1

                                    # cv2.rectangle(frame, (int(detection.x), int(detection.y)),
                                    #               (int(detection.x + detection.w), int(detection.y + detection.h)),
                                    #               (255, 255, 255), 2)

                                cv2.putText(background, timestamp, (int(detection.x - multiplier * detection.w), int(detection.y - multiplier * h + 16)), 0, 3e-3 * 200, (0,), 4)
                                cv2.putText(background, timestamp, (int(detection.x - multiplier * detection.w), int(detection.y - multiplier * h + 16)), 0, 3e-3 * 200, (255, 255, 255), 1)

                                # cv2.putText(background, timestamp, (int(detection.x), int(detection.y)), 0,
                                #             3e-3 * 210, (0, 0, 0), 1, cv2.LINE_AA)
                                # cv2.putText(background, timestamp, (int(detection.x), int(detection.y)), 0,
                                #             3e-3 * 200, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('gen', background)

            if writeVideo_flag:
                out.write(background)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break


        # while True:
        #
        #     image_stack = []
        #     # background = None
        #
        #     for gen_idx, gen in enumerate(gens):
        #         # frame, det, timestamp = next(gen)
        #
        #         image_stack.append(next(gen))
        #
        #     image_stack.sort(key=lambda x: x[3])
        #
        #     for gen_idx, (frame, det, timestamp, pos) in enumerate(image_stack):
        #         if gen_idx == 0:
        #             background = frame
        #         if frame is not None:
        #             if det:
        #                 for detection in det:
        #                     # cv2.rectangle(frame, (int(detection.x), int(detection.y)),
        #                     #               (int(detection.x + detection.w), int(detection.y + detection.h)),
        #                     #               (255, 255, 255), 2)
        #
        #                     x = detection.x
        #                     y = detection.y
        #                     w = detection.w
        #                     h = detection.h
        #
        #                     # TODO можно сортировать события по времени перед наложением
        #                     # так чтобы самые новые события были поверх более старых
        #                     # background[y:y + h, x:x + w] = frame[y:y + h, x:x + w]
        #                     cv2.addWeighted(frame[y:y + h, x:x + w], 0.7, background[y:y + h, x:x + w], 0.3, 0, background[y:y + h, x:x + w])
        #
        #
        #                 for detection in det:
        #                     cv2.putText(background, timestamp, (int(detection.x), int(detection.y)), 0,
        #                                 3e-3 * 200, (0, 255, 0), 1, cv2.LINE_AA)
        #     cv2.imshow('gen', background)
        #
        #     if writeVideo_flag:
        #         out.write(background)
        #
        #     k = cv2.waitKey(1) & 0xff
        #     if k == 27:
        #         break


    # writer.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    events = collect_events()
    main(events)
