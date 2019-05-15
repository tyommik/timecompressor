from collections import namedtuple
import sorting
import cv2
import numpy as np
from itertools import combinations
from enum import Enum, auto

from events import Event
from video_processing import VideoCapture, channel_generator

CONST_TIME = 539975
Detection = namedtuple('Detection', 'pos, x, y, w, h, obj_class, precision')

video_input = r'/home/ashibaev/Downloads/timecompressor_archive.mp4'
video_output = r'summary_7.mp4'
cap = cv2.VideoCapture(video_input)
cap.set(2, 1500)

writeVideo_flag = True


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
            video = VideoCapture(video_source=video_input, start_frame=event.start, shift_time=CONST_TIME)

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

            for w in working_channels:
                w.run()

            image_stack = []

            list_of_changable_gens = []

            for gen_idx, gen in enumerate([ch for ch in working_channels_gens]):
                # frame, det, timestamp = next(gen)

                tp = next(gen)
                if tp is Op.STOP:
                    # запомни индекс генератора, который бросил остановку
                    list_of_changable_gens.append(gen_idx)
                else:
                    image_stack.append(tp)

            # обработать list_of_changable_gens - список остановленных генераторов

            new_working_channels = []
            new_working_channels_gens = []

            for idx, w in enumerate(working_channels):
                # если генератор не в списке, то оставляем его в массиве работающих
                if idx not in list_of_changable_gens:
                    new_working_channels.append(working_channels[idx])
                    new_working_channels_gens.append(working_channels_gens[idx])
                else:
                    # иначе нужно добавить новый в замен старого
                    print('Создание нового генератора, добавление в очередь ждущих')
                    new_ch = VideoChannel(events)
                    waiting_channels.append(new_ch)
                    waiting_channels_gens.append(new_ch.step())

            working_channels[:] = new_working_channels
            working_channels_gens[:] = new_working_channels_gens

            if image_stack:
                # отсортируем по времени
                image_stack.sort(key=lambda x: x[3])

                pass_or_not = False

                for gen_idx, (frame, det, timestamp, pos) in enumerate(image_stack):
                    if gen_idx == 0:
                        background = frame
                    if frame is not None:
                        if det:
                            for detection in det:
                                if waiting_channels:
                                    check_channel, check_channel_gen = waiting_channels.pop(
                                        0), waiting_channels_gens.pop(0)

                                    _, w_dets, *args = next(check_channel_gen)

                                    for w_det in w_dets:
                                        pass_or_not = True

                                        res = sorting.check_rectangels(w_det, detection)
                                        if res <= 20:
                                            print('Площадь пересчения меньше 20%')
                                            pass_or_not = True  # pass_or_not * True
                                        else:
                                            pass
                                            # print('Площадь пересечения равна ', res)

                                    if pass_or_not:

                                        print('Переключаем генератор из ждущего в работающий')

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

                                multiplier = 0.0

                                assert y - int(multiplier * h) > 0 or x - int(multiplier * w)

                                cv2.addWeighted(frame[y - int(multiplier * h): y + h + int(multiplier * h),
                                                x - int(multiplier * w): x + w + int(multiplier * w)], 0.7,
                                                background[y - int(multiplier * h): y + h + int(multiplier * h),
                                                x - int(multiplier * w): x + w + int(multiplier * w)], 0.3, 0,
                                                background[y - int(multiplier * h): y + h + int(multiplier * h),
                                                x - int(multiplier * w): x + w + int(multiplier * w)])

                            for detection in det:
                                multiplier = 0.0

                                cv2.putText(background, timestamp, (
                                int(detection.x - multiplier * detection.w), int(detection.y - multiplier * h)), 0,
                                            3e-3 * 200, (0,), 4)
                                cv2.putText(background, timestamp, (
                                int(detection.x - multiplier * detection.w), int(detection.y - multiplier * h)), 0,
                                            3e-3 * 200, (255, 255, 255), 1)

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

    # writer.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    events = collect_events()
    main(events)