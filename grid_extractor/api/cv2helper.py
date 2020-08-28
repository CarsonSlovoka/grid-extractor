import numpy as np
from typing import Union, List, Callable
import ctypes
import cv2


def show_img(img_list: Union[np.ndarray, List[np.ndarray]], combine_fun: Callable = np.vstack,
             window_name='demo', window_size=(ctypes.windll.user32.GetSystemMetrics(0) // 2, ctypes.windll.user32.GetSystemMetrics(1) // 2),
             delay_time=0, note: Union[str, List[str]] = None, **options):
    if isinstance(img_list, np.ndarray):
        img_list = [img_list]

    if isinstance(note, str):
        print(note)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if window_size:
        w, h = window_size
        cv2.resizeWindow(window_name, w, h)

    result_list = []
    for idx, img in enumerate(img_list):
        img = np.copy(img)
        if len(img.shape) != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            print('convert gray to bgr') if options.get('debug') else None
        if note and isinstance(note, list) and idx < len(note):
            org_x, org_y = org = options.get('org', (50, 50))
            img = cv2.copyMakeBorder(img, org_y*2, 0, 0, 0,
                                     cv2.BORDER_CONSTANT, value=(128, 128, 128))
            cv2.putText(img, note[idx], org=org,
                        fontFace=options.get('fontFace', cv2.FONT_HERSHEY_SIMPLEX),
                        fontScale=options.get('fontScale', 2),
                        color=options.get('color', (0, 255, 255)),
                        thickness=options.get('color', 4))
        result_list.append(img)
    cv2.imshow(window_name, combine_fun(result_list))
    cv2.waitKey(delay_time)
