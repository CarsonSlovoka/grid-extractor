import cv2
import numpy as np
from .api.cv2helper import show_img
from typing import Callable, Generator, Iterator, Tuple
from pathlib import Path
import functools


class GridExtractorBase:
    __slots__ = ('img_bgr',)

    def __init__(self, img_path: Path):
        if not img_path.exists():
            raise FileNotFoundError(img_path.absolute())
        self.img_bgr = cv2.imread(str(img_path))

    @property
    def channel(self):
        _h, _w, *channel = self.img_bgr.shape
        return channel[0] if channel else 1

    def _get_hv_contours(self, kernel_size: Tuple[int, int], iterations, **options):
        """
        :param kernel_size:  (x, y)
        :param iterations: The number of points obtained is inversely proportional to this value.
        :return:
        """
        gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        img_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # img_thresh = cv2.dilate(img_thresh, np.ones((1, 40)))
        if options.get('morph_close'):
            img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel, iterations=iterations)  # If the line is not clear, we can make it more clear.
        img_detect = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel, iterations=iterations)
        contours, hierarchy = cv2.findContours(img_detect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        show_img([img_thresh, img_detect], combine_fun=np.hstack) if options.get('debug') else None
        for c in contours:
            """
            x, y, w, h = cv2.boundingRect(c)
            area: float = cv2.contourArea(c)
            rect: np.ndarray = cv2.minAreaRect(c)
            box: np.ndarray = cv2.boxPoints(rect)
            """
            yield c  # cv2.drawContours(clean, [c], -1, color=, thickness=)

    def get_horizontal_contours(self, kernel_size: Tuple[int, int], iterations=1, **options):
        """
        kernel_size:  (15, 1)
        """
        yield from self._get_hv_contours(kernel_size, iterations, **options)

    def get_vertical_contours(self, kernel_size: Tuple[int, int], iterations=1, **options):
        """
        kernel_size:  (1, 15)
        """
        yield from self._get_hv_contours(kernel_size, iterations, **options)

    def get_hough_lines(self, **options):
        img_gray: np.ndarray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
        img_bit_edges: np.ndarray = cv2.Canny(img_gray, 183, 153, apertureSize=3)
        show_img(img_bit_edges)

        lines_p = cv2.HoughLinesP(img_bit_edges, rho=1,
                                  theta=options.get('theta', np.pi / 180),
                                  threshold=options.get('threshold', 1), lines=options.get('lines'),
                                  minLineLength=options.get('minLineLength', 100),
                                  maxLineGap=options.get('maxLineGap', 3)
                                  )
        if lines_p is not None:
            print(f'A total of {lines_p.shape[0]} line segments found.')
            for line in lines_p:
                for x1, y1, x2, y2 in line:
                    yield (x1, y1), (x2, y2)   # cv2.line(img, pt1, pt2, (255, 255, 255), 2)

    def get_contours(self, **options) -> Iterator[Tuple[
        Tuple[int, int, int, int],
        float,
        np.ndarray,
        np.ndarray
    ]]:
        """
        :param options:
            threshold_fun = lambda img_gray: cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)  # np.where(img_gray > 127, 255, 0)  # bit
            dict_morph_open=dict(kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            dict_hv_line=dict(blur_value=30, h_ratio=1.2, w_ratio=1.2)
        :return:
        """
        img_gray = cv2.cvtColor(options.get('img', self.img_bgr), cv2.COLOR_BGR2GRAY)

        dict_morph_open = options.get('dict_morph_open', dict())
        if dict_morph_open:
            img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN,
                                        kernel=dict_morph_open.get('kernel', cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
                                        )  # make the border is more clear

        img_gray = 255 - img_gray

        threshold_val, img_thresh = functools.wraps(cv2.threshold)(options['threshold_fun'])(img_gray) if options.get('threshold_fun') else \
            cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # grayscale

        if options.get('debug'):
            show_img([img_thresh], note=['img_thresh'])

        contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.RETR_CCOMP)
        for i, c in enumerate(contours):
            x, y, w, h = cv2.boundingRect(c)
            area: float = cv2.contourArea(c)
            # ratio_w_h = w / h
            rect: np.ndarray = cv2.minAreaRect(c)
            box: np.ndarray = cv2.boxPoints(rect)
            yield (x, y, w, h), area, rect, box


class ShadowClearMixin:
    __slots__ = ()

    @staticmethod
    def analysis_by_channel(img: np.ndarray, **options):
        """
        :param img
        :param options:
            - dilate_kernel
            - median_blur_size
            - combine_fun
        :return:
        """
        dilate_kernel = options.get('dilate_kernel', (7, 7))
        median_blur_size = options.get('median_blur_size', 21)
        result = []
        result_norm = []
        for i, img_channel in enumerate(cv2.split(img)):
            img_channel: np.ndarray
            img_dilate = cv2.dilate(img_channel, np.ones(dilate_kernel, np.uint8))
            img_bg = cv2.medianBlur(img_dilate, median_blur_size)
            diff = cv2.absdiff(img_channel, img_bg)
            diff_inv = 255 - diff

            img_norm: np.ndarray = cv2.normalize(diff_inv, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

            show_img([img_channel, img_bg, diff, diff_inv, img_norm], combine_fun=options.get('combine_fun', np.vstack),
                     note=[f'channel:{i}', 'bg', 'diff', 'diff inv', 'img_norm'],
                     debug=True)

            result.append(diff_inv)
            result_norm.append(img_norm)
        img_combine = cv2.merge(result)
        img_combine_norm = cv2.merge(result_norm)
        show_img([img_combine, img_combine_norm], note=['combine', 'combine norm'])
        return img_combine, img_combine_norm
