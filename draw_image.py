import cv2
import numpy as np


def draw_time_axis(start_time: int, len_max: int, one_track_wide=16) -> np.ndarray:
    track_time = np.ones((one_track_wide, len_max, 3), dtype=np.uint8) * 255
    for i in range(track_time.shape[1]):
        if (i + start_time) % 60 == 0:
            track_time[-1 * int(one_track_wide * 0.25):, i] = (0, 0, 128)
        if (i + start_time) % 600 == 0:
            track_time[-1 * int(one_track_wide * 0.75):, i] = (0, 0, 128)
            cv2.putText(track_time, str((i + start_time) // 60), (i + 2, track_time.shape[0] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 128))
    return track_time


def draw_track_map(sequence_msg: list, labels: (list, tuple), start_time: int,
                   one_track_wide=16, have_time_track=True) -> np.ndarray:
    len_max = len(sequence_msg)
    heatmap = np.ones((one_track_wide * len(labels), len_max, 3), dtype=np.uint8) * 255

    for i, label in enumerate(sequence_msg):
        if label == -1:
            continue
        heatmap[label * one_track_wide:(label + 1) * one_track_wide, i] = (0, 216, 0)
    heatmap = heatmap[::-1]

    time_track = draw_time_axis(start_time, len_max, one_track_wide)
    if have_time_track:
        heatmap = np.concatenate((heatmap, time_track), 0)
    else:
        heatmap = np.concatenate((heatmap, np.zeros((8, heatmap.shape[1], 3), dtype=np.uint8)), 0)

    num_interval = len(labels) + 1
    for i in range(num_interval):
        heatmap[i * one_track_wide - 1:i * one_track_wide + 2] = (128, 128, 128)
    return heatmap


def draw_confidence_line(confidence_values, tf=None):
    colors = ((0, 0, 218), (0, 218, 0), (0, 218, 218), (255, 255, 255))

    if tf is None:
        tf = [1, ] * len(confidence_values)
    background = np.ones((100, len(confidence_values), 3), dtype=np.uint8) * 255
    for i in [3, 5, 7]:
        cv2.line(background, (0, i * 10), (len(confidence_values) - 1, i * 10), (64, 64, 64))

    last_set = (-1, -1)
    for x, cfg_value in enumerate(confidence_values):
        if cfg_value == -1:
            last_set = (-1, -1)
            continue
        if sum(last_set) == -2:
            now_set = (x, 100 - int(cfg_value * 100))
            cv2.circle(background, now_set, 1, colors[tf[x]], -1)
            last_set = now_set
        else:
            now_set = (x, 100 - int(cfg_value * 100))
            cv2.circle(background, now_set, 1, colors[tf[x]], -1)
            cv2.line(background, last_set, now_set, colors[2])
            last_set = now_set

    confidence_map = np.concatenate((background, np.zeros_like(background, dtype=np.uint8)[:4]), 0)
    return confidence_map


def norm_matrix_2_rgb_heatmap(norm_map: np.ndarray, mode="BGR"):
    if len(norm_map.shape) != 3 or norm_map[2] != 3:
        raise ValueError("norm_map dim is 3, and shape is H,W,3")

    norm_map[norm_map > 1] = 1
    norm_map[norm_map < 0] = 0

    norm_map = np.expand_dims(norm_map, 2).repeat(3, 2)
    min_values = np.min(norm_map)
    max_values = np.max(norm_map)
    norm_map = (norm_map - min_values) / (max_values - min_values)

    if mode == "BGR":
        norm_map[..., 2] = norm_map[..., 2] * 510 - 255
        norm_map[..., 1] = (-4 * norm_map[..., 1] * norm_map[..., 1] + 4 * norm_map[..., 1]) * 255
        norm_map[..., 0] = norm_map[..., 0] * (-510) + 255
        norm_map[norm_map < 0] = 0
        norm_map[norm_map > 255] = 255
    elif mode == "gray":
        norm_map[...] = norm_map[...] * 510 - 255
        norm_map[norm_map < 0] = 0
        norm_map[norm_map > 255] = 255
    return norm_map


def draw_color_histogram_with_image(image: np.ndarray, formula_mode="AVG"):
    histogram = np.zeros((100, 256, 3), dtype=np.uint8)
    channel_sorted = ["B", "G", "R"]
    reture_value = {"B": 0,
                    "G": 0,
                    "R": 0}

    for channel_id in range(3):
        color_values_statistic = np.unique(image[..., channel_id], return_counts=True)
        channel_count_max = np.max(color_values_statistic[1])
        for i in range(color_values_statistic[0].shape[0]):
            color_values = color_values_statistic[0][i]
            color_values_percent = round(color_values_statistic[1][i] / channel_count_max * 100)
            histogram[100 - color_values_percent:, color_values, channel_id] = 218
        if formula_mode == "AVG":
            reture_value[channel_sorted[channel_id]] = np.average(image[..., channel_id]).astype(np.float16)

    image[:100, image.shape[1] - 256:] = histogram

    if formula_mode is None:
        return image
    else:
        return image, reture_value
