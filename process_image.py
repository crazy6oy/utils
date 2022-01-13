import base64

import cv2
import numpy as np


def str_2_image(image_ascii_data: str) -> np.uint8:
    image_ascii_data = image_ascii_data.encode('ascii')
    image_byte = base64.b64decode(image_ascii_data)
    image = cv2.imdecode(np.asarray(bytearray(image_byte), dtype='uint8'), cv2.IMREAD_COLOR)  # for jpg

    return image


def image_2_str(image):
    _, image_encode = cv2.imencode('.jpg', image)
    image_bytes = image_encode.tostring()
    image_base64_data = base64.b64encode(image_bytes)
    image_ascii_data = image_base64_data.decode("ascii")

    return image_ascii_data

def drawTrackmap(sequence_msg, labels, start_time, have_time_track=True):
    """
    生成识别结果轨道图

    :params sequence_msg:类型（list-int）结果数据，背景为0，其他的类别转换成从1、2、3...类别id
    :params labels:类型（list-int）类型ID列表，sequenceMsg中出现的ID都在在这出现，但只出现一次
    :params start_time:类型（int），序列第一个数据是第多少秒的结果
    :params have_time_track:类型（bool）是否在轨道下面显示时间轨道
    :return 类型(np.array-uint8)BGR类别轨道图
    """
    one_track_wide = 16
    len_max = len(sequence_msg)

    heatmap = np.ones((one_track_wide * len(labels), len_max, 3), dtype=np.uint8) * 255
    for i, label in enumerate(sequence_msg):
        if label == -1:
            continue
        label_index = label
        heatmap[label_index * one_track_wide:(label_index + 1) * one_track_wide, i] = (0, 216, 0)
    heatmap = heatmap[::-1]

    track_time = np.ones((one_track_wide, len_max, 3), dtype=np.uint8) * 255
    for i in range(track_time.shape[1]):
        if (i + start_time) % 60 == 0:
            track_time[-1 * int(one_track_wide * 0.25):, i] = (0, 0, 128)
        if (i + start_time) % 600 == 0:
            track_time[-1 * int(one_track_wide * 0.75):, i] = (0, 0, 128)
            cv2.putText(track_time, str((i + start_time) // 60), (i + 2, track_time.shape[0] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 128))
    if have_time_track:
        heatmap = np.concatenate((heatmap, track_time), 0)
    else:
        heatmap = np.concatenate((heatmap, np.zeros((8, heatmap.shape[1], 3), dtype=np.uint8)), 0)

    num_interval = len(labels) + 1
    for i in range(num_interval):
        heatmap[i * one_track_wide - 1:i * one_track_wide + 2] = (128, 128, 128)
    return heatmap


if __name__ == '__main__':
    import json

    json_path = r"Z:\withai\dataset\processed-data\18-instrument-box-v0\processed\LC-CSR-50-5-0455.json"
    with open(json_path) as f:
        msg = json.load(f)
    img_data = msg["imageData"]
    img = str_2_image(img_data)
    ss = image_2_str(img)
    stop = 0
