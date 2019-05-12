from collections import namedtuple


def check_rectangels(det_a, det_b):
    """
    Check two rectangels, use TOP-LEFT BOTTOM-RIGHT scheme

    :param rectangle_one:
    :param rectangel_two:
    :return: overlap precentage of overlapped_area/ra_area
    """

    # Для проверки
    rectangle_one = (det_a.x, det_a.y, det_a.x + det_a.w, det_a.y + det_a.h)
    rectangle_two = (det_b.x, det_b.y, det_b.x + det_b.w, det_b.y + det_b.h)

    def area(a, b):  # returns None if rectangles don't intersect
        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
        if (dx >= 0) and (dy >= 0):
            return dx * dy

    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

    ra = Rectangle(*rectangle_one)
    rb = Rectangle(*rectangle_two)

    overlapped_area = area(ra, rb)

    ra_area = (ra.xmax - ra.xmin) * (ra.ymax - ra.ymin)
    rb_area = (rb.xmax - rb.xmin) * (rb.ymax - rb.ymin)

    if overlapped_area is None:
        return 0

    overlap_percentage = round((overlapped_area / ra_area) * 100, 2)

    return overlap_percentage


def search_optimal(event_a, event_b):

    result = []

    our_range = abs(event_a.length - event_b.length)
    shorter_event, longer_event = sorted([event_a, event_b], key=lambda x: x.length)

    for shift in range(our_range):
        r = []

        for i in range(shorter_event.stop-shorter_event.start):
            event_a_step, event_b_step = longer_event.start + i, shorter_event.start + i + shift
            det_a_list = longer_event.detections_map.get(event_a_step, [])
            det_b_list = shorter_event.detections_map.get(event_b_step, [])

            local_result = 0

            for det_in_a in det_a_list:
                for det_in_b in det_b_list:
                    local_result += check_rectangels(det_in_a, det_in_b)

            r.append(local_result)

        result.append((sum(map(int, r)), shift))

    return min(result, key=lambda x: x[0])[1]






