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


if __name__ == '__main__':
    import json

    json_path = r"Z:\withai\dataset\processed-data\18-instrument-box-v0\processed\LC-CSR-50-5-0455.json"
    with open(json_path) as f:
        msg = json.load(f)
    img_data = msg["imageData"]
    img = str_2_image(img_data)
    ss = image_2_str(img)
    stop = 0
