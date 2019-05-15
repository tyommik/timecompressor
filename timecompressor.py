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
    WIDTH, HIGHT = int(cap.get(3)), int(cap.get(4))

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(video_output, fourcc, 25, (WIDTH, HIGHT))

    channels_number = 4

    counter = 0

    while events:

        background = None

        working_channels = []
        waiting_channels = [VideoChannel(events) for _ in range(channels_number)]

        working_channels.append(waiting_channels.pop(0))

        for w in working_channels:
            w.run()

        working_channels_gens = [ch.step() for ch in working_channels]
        waiting_channels_gens = [ch.step() for ch in waiting_channels]

        while True:

            if not working_channels:
                working_channels.append(waiting_channels.pop(0))
                working_channels_gens.append(waiting_channels_gens.pop(0))

            while(len(waiting_channels) + len(working_channels) < channels_number):
                new_ch = VideoChannel(events)
                waiting_channels.append(new_ch)
                waiting_channels_gens.append(new_ch.step())

            # обязательно переключаем все каналы в работающие
            for w in working_channels:
                w.run()

            image_stack = []

            list_of_changable_gens = []

            for gen_idx, gen in enumerate([ch for ch in working_channels_gens]):
                tp = next(gen)
                if tp is Op.STOP:
                    # запомни индекс генератора, который бросил остановку
                    print('Генератор прекратир работу')
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

                active_detections = []
                for image in image_stack:
                    active_detections += image[1]

                if waiting_channels:

                    for _ in range(len(waiting_channels)):
                        check_channel, check_channel_gen = waiting_channels.pop(
                            0), waiting_channels_gens.pop(0)

                        _, w_dets, *args = next(check_channel_gen)

                        if w_dets:

                            pass_or_not = True

                            for w_det in w_dets:
                                for active_detection in active_detections:
                                    res = sorting.check_rectangels(w_det, active_detection)
                                    if res <= 5:
                                        pass_or_not *= True
                                    else:
                                        pass_or_not *= False

                            if pass_or_not:

                                print('Переключаем генератор из ждущего в работающий')

                                working_channels.append(check_channel)
                                working_channels_gens.append(check_channel_gen)
                                break

                            else:
                                # waiting_channels.insert(0, check_channel)
                                # waiting_channels_gens.insert(0, check_channel_gen)

                                print('Событие не поместилось в кадр')
                                waiting_channels.append(check_channel)
                                waiting_channels_gens.append(check_channel_gen)

                for gen_idx, (frame, det, timestamp, pos) in enumerate(image_stack):

                    if gen_idx == 0:
                        background = frame
                    if det and frame is not None:
                        for detection in det:

                            w = detection.w
                            h = detection.h

                            w_additive = int((0.8 * w) / 2)
                            h_additive = int((0.2 * h) / 2)

                            if detection.x - w_additive >= 0:
                                x = detection.x - w_additive
                            else:
                                x = 0

                            if detection.x + w + w_additive <= WIDTH:
                                w = w + w_additive
                            else:
                                w = WIDTH

                            if detection.y - h_additive >= 0:
                                a = detection.y
                                y = detection.y - h_additive
                            else:
                                y = 0

                            if detection.y + h + h_additive <= HIGHT:
                                h = detection.y + h + h_additive
                            else:
                                h = HIGHT

                            cv2.addWeighted(frame[y: y + h,
                                            x: x + w], 0.7,
                                            background[y: y + h,
                                            x: x + w], 0.3, 0,
                                            background[y: y + h,
                                            x: x + w])

                        # for detection in det:
                            cv2.rectangle(background, (int(x), int(y)),
                                          (int(x + w), int(y + h)),
                                          (242, 255, 229), 1)

                            cv2.putText(background, timestamp, (
                            int(x + 4), int(y + 12)), 0,
                                        3e-3 * 120, (0,), 4)
                            cv2.putText(background, timestamp, (
                            int(x + 4), int(y + 12)), 0,
                                        3e-3 * 120, (255, 255, 255), 1)

            cv2.imshow('gen', background)

            counter += 1
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