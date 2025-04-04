import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from word_detector import detect, prepare_img, sort_multiline
from pathlib import Path
from typing import List

list_img_names_serial = []


def get_img_files(data_dir: Path) -> List[Path]:
    """Return all image files contained in a folder."""
    res = []
    for ext in ['*.png', '*.jpg', '*.bmp']:
        res += list(Path(data_dir).glob(ext))
    return res


def save_image_names_to_text_files(data_path: str, img_height: int = 1000, kernel_size: int = 25, sigma: float = 11.0, theta: float = 7.0, min_area: int = 100):
    data_dir = Path(data_path)
    path = './test_images'
    os.makedirs(path, exist_ok=True)

    for fn_img in get_img_files(data_dir):
        print(f'Processing file {fn_img}')

        img = prepare_img(cv2.imread(str(fn_img)), img_height)
        detections = detect(img, kernel_size=kernel_size, sigma=sigma, theta=theta, min_area=min_area)
        lines = sort_multiline(detections)

        plt.imshow(img, cmap='gray')
        colors = plt.cm.get_cmap('rainbow', 7)

        for line_idx, line in enumerate(lines):
            for word_idx, det in enumerate(line):
                xs = [det.bbox.x, det.bbox.x, det.bbox.x + det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
                ys = [det.bbox.y, det.bbox.y + det.bbox.h, det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]
                plt.plot(xs, ys, c=colors(line_idx % 7))
                plt.text(det.bbox.x, det.bbox.y, f'{line_idx}/{word_idx}')

                crop_img = img[det.bbox.y:det.bbox.y + det.bbox.h, det.bbox.x:det.bbox.x + det.bbox.w]
                img_name = f'line{line_idx}_word{word_idx}.jpg'
                cv2.imwrite(os.path.join(path, img_name), crop_img)

                list_img_names_serial.append(img_name)
                print(list_img_names_serial)

        with open('./examples/img_names_sequence.txt', 'w') as textfile:
            for element in list_img_names_serial:
                textfile.write(element + '\n')

        plt.show()


save_image_names_to_text_files('./figures')