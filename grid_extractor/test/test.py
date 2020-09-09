import unittest
from pathlib import Path
import cv2
import numpy as np
from types import ModuleType
import inspect

if 'env path':
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from grid_extractor import __version__
    from grid_extractor.core import (
        ShadowClearMixin, GridExtractorBase, OCRMixin,
        show_img
    )

    sys.path.remove(sys.path[0])


class GB18030Tests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        class GB18030Factory(GridExtractorBase, ShadowClearMixin, OCRMixin):
            ...

        self.obj = GB18030Factory(Path('image/GB18030/no_watermark.jpg'))

    def test_analysis_by_channel(self):
        self.obj.analysis_by_channel(self.obj.img_bgr)

    def test_get_contours(self):
        """
        for pt1, pt2 in self.obj.get_hough_lines():
            cv2.line(self.obj.img_bgr, pt1, pt2, (255, 0, 255), 2)

        show_img(self.obj.img_bgr)
        """

        v_contours_list = [cv2.boundingRect(c) for c in self.obj.get_vertical_contours((1, 100), debug=True)]
        h_contours_list = [cv2.boundingRect(c) for c in self.obj.get_horizontal_contours((100, 1), debug=True)]

        img_bgr_copy = np.copy(self.obj.img_bgr)
        for x, y, w, h in v_contours_list + h_contours_list:
            cv2.rectangle(img_bgr_copy, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)

        show_img(img_bgr_copy)
        img_bgr_copy_1 = np.copy(img_bgr_copy)
        img_bgr_copy_2 = np.copy(img_bgr_copy)
        for (x, y, w, h), area, rect, box in self.obj.get_contours(
            img=img_bgr_copy,
            # threshold_fun=lambda img_gray: cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY),
            dict_morph_open=dict(on=True, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (10, 4))),
            debug=True
        ):
            if 75 < area < 1200 and w < 30 and h > 15:
                # cv2.rectangle(img_bgr_copy_1, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
                # cv2.drawContours(img_bgr_copy_2, [np.int0(box)], -1, color=(255, 0, 0), thickness=2)

                img_char = self.obj.img_bgr[y - 2:y + h, x - 2:x + w]
                img_id = self.obj.img_bgr[y + h + 7: y + h + 20, x - 8:x + w + 7]
                """
                img_id_gray = cv2.cvtColor(img_id, cv2.COLOR_BGR2GRAY)
                img_id_gray = cv2.threshold(img_id_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                img_id_gray[np.where(np.count_nonzero(img_id_gray, axis=1) == 0)[0]] = 255  # let the black line (which is all black of each column) to the white color.
                self.obj.ocr_predict(img_id_gray, '--psm=6')
                """
                img_temp_combine = np.ones((img_char.shape[0] + img_id.shape[0],
                                            max(img_char.shape[1], img_id.shape[1]), self.obj.channel),
                                           dtype=np.uint8) * 128

                img_temp_combine[:img_char.shape[0], :img_char.shape[1]] = img_char
                img_temp_combine[img_char.shape[0]:img_char.shape[0] + img_id.shape[0], :img_id.shape[1]] = img_id
                # cv2.imwrite('', img)
                id_string = self.obj.ocr_predict(cv2.resize(img_id, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC),
                                                 # --oem 0
                                                 config='--psm 6 -c tessedit_char_whitelist=0123456789ABCDEF -c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyzGHIJKLMNOPQRSTUVWXYZ')
                print(id_string.strip())
                # show_img(img_temp_combine, window_size=(40, 40))
                cv2.rectangle(self.obj.img_bgr, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

        # show_img([img_bgr_copy_1, img_bgr_copy_2])
        show_img(self.obj.img_bgr)


class JISGridTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        class JISFactory(GridExtractorBase, ShadowClearMixin):
            ...

        self.obj = JISFactory(Path('image/JIS/jis_x_ch_pg_08.tiff'))

    """
    def test_analysis_by_channel(self):
        self.obj.analysis_by_channel(self.obj.img_bgr)
    """

    def test_get_contours(self):
        h_pt_list = []
        debug = False
        for c in self.obj.get_horizontal_contours((1000, 1), iterations=1, debug=debug):
            x, y, w, h = cv2.boundingRect(c)
            # cv2.drawContours(self.obj.img_bgr, [np.int0(box)], -1, color=(255, 255, 255), thickness=2)
            # cv2.rectangle(self.obj.img_bgr, (x, y), (x+w, y + h), (255, 255, 255), thickness=8)
            h_pt_list.append((x, y, w, h))

        v_pt_list = []
        for c in self.obj.get_vertical_contours((1, 1000), iterations=1, debug=debug):
            x, y, w, h = cv2.boundingRect(c)
            v_pt_list.append((x, y, w, h))

        for x, y, w, h in h_pt_list + v_pt_list:
            cv2.rectangle(self.obj.img_bgr, (x, y), (x + w, y + h), (255, 255, 255), thickness=8)

        img_bgr_copy_1 = np.copy(self.obj.img_bgr)
        img_bgr_copy_2 = np.copy(self.obj.img_bgr)
        area_list = []
        for (x, y, w, h), area, rect, box in self.obj.get_contours(
            # threshold_fun=lambda img_gray: cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY),
            dict_morph_open=dict(kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))),
            debug=True
        ):
            if 500 < area < 6000:
                # show_img(self.obj.img_bgr[y: y + h, x: x + w, :])
                cv2.rectangle(img_bgr_copy_1, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
                cv2.drawContours(img_bgr_copy_2, [np.int0(box)], -1, color=(255, 0, 0), thickness=2)
                area_list.append(area)
        print(f'area average:{sum(area_list) / len(area_list)}')
        show_img([img_bgr_copy_1, img_bgr_copy_2], combine_fun=np.hstack)
        show_img(img_bgr_copy_1)
        # view_min_y, view_min_x = img_bgr_copy_1.shape[0]//2, img_bgr_copy_1.shape[1]//2
        # show_img([img_bgr_copy_1[view_min_y:, view_min_x:]], combine_fun=np.hstack)


class JISNoGridTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

        class JISNoGridFactory(GridExtractorBase):
            ...

        self.obj = JISNoGridFactory(Path('image/JIS/jis_x_j_pr10_i4_pg_229.tiff'))
        """
        import matplotlib.pyplot as plt
        img_rgb = self.obj.img_bgr[:, :, ::-1]
        plt.imshow(img_rgb.astype(int))
        plt.show()
        """

    def test_get_contours(self):
        img_bgr_copy_1 = np.copy(self.obj.img_bgr)
        debug = False

        # remove horizontal line
        for c in self.obj.get_horizontal_contours((1000, 1), morph_close=True, iterations=1, debug=debug):
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img_bgr_copy_1, (x, y), (x + w, y + h), (255, 255, 255), thickness=8)

        area_list = []
        for (x, y, w, h), area, rect, box in self.obj.get_contours(
            img=img_bgr_copy_1,
            # threshold_fun=lambda img_gray: cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY),
            dict_morph_open=dict(kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))),
            debug=debug
        ):
            if 16000 < area < 26000 and 150 < w < 200:
                # print(w)
                # show_img(self.obj.img_bgr[y:y + h, x:x + w])
                # show_img(self.obj.img_bgr[y:y + h // 3 + 10, x - 600:x - 600 + 450])
                cv2.rectangle(self.obj.img_bgr, (x, y), (x + w, y + h), (0, 255, 0), thickness=5)
                area_list.append(area)
        print(f'max(area):{max(area_list)}')
        print(f'area average:{sum(area_list) / len(area_list)}')
        show_img([self.obj.img_bgr], combine_fun=np.hstack)
        # view_min_y, view_min_x = img_bgr_copy_1.shape[0]//2, img_bgr_copy_1.shape[1]//2
        # show_img([self.obj.img_bgr[view_min_y:, view_min_x:]], combine_fun=np.hstack)


class GSCCTests(unittest.TestCase):
    """
    General Standard Chinese Character
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        class GSCCFactory(GridExtractorBase):
            ...

        self.obj = GSCCFactory(Path('image/GeneralStandardChineseCharacter/pg_045.jpg'))

    def test_get_contours(self):
        img_bgr_copy = np.copy(self.obj.img_bgr)
        for c in self.obj.get_vertical_contours(
            (1, 500),
            # debug=True
        ):
            cv2.drawContours(img_bgr_copy, [c], -1, color=(255, 255, 255), thickness=10)

        _h, _w, *channel = self.obj.img_bgr.shape
        channel = channel[0] if channel else 1
        area_list = []
        w_list = []
        h_list = []
        for (x, y, w, h), area, rect, box in self.obj.get_contours(
            img=img_bgr_copy,
            dict_morph_open=dict(kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (35,  # w
                                                                                   20)  # h
                                                                  )),
            debug=True
        ):
            if 4000 < area < 20000 and 150 < w < 220 and 50 < h < 95:
                area_list.append(area)
                w_list.append(w)
                h_list.append(h)
                print(area, w, h)

                img_char = self.obj.img_bgr[y - 50:y + h + 25, x + 240:x + 240 + 140]
                img_id = self.obj.img_bgr[y:y + h, x:x + w]
                img_temp_combine = np.ones((max(img_char.shape[0], img_id.shape[0]),
                                            img_char.shape[1] + img_id.shape[1], channel),
                                           dtype=np.uint8) * 128
                img_temp_combine[:img_id.shape[0], :img_id.shape[1]] = img_id
                img_temp_combine[:img_char.shape[0], img_id.shape[1]:img_id.shape[1] + img_char.shape[1]] = img_char
                # show_img(img_temp_combine)

                cv2.rectangle(self.obj.img_bgr, (x, y), (x + w, y + h), (0, 255, 0), thickness=5)
        print(min(w_list), max(w_list))
        print(min(h_list), max(h_list))
        print(min(area_list), max(area_list))
        show_img(self.obj.img_bgr)


class CLITests(unittest.TestCase):

    def test_show_version(self):
        print(__version__)
        self.assertTrue(len(__version__) > 0)


def test_setup():
    # suite_list = [unittest.TestLoader().loadTestsFromTestCase(class_module) for class_module in (CLITests, )]
    # suite_class_set = unittest.TestSuite(suite_list)

    suite_function_set = unittest.TestSuite()
    suite_function_set.addTest(CLITests('test_show_version'))

    suite = suite_function_set  # pick one of two: suite_class_set, suite_function_set
    # unittest.TextTestRunner(verbosity=1).run(suite)  # self.verbosity = 0  # 0, 1, 2.  unittest.TextTestResult
    return suite


def run_all_tests_case(module: ModuleType):
    """
    It's better to run this function if you modify old source code, to check all services have exceptional.

    USAGE:

        1. run_all_tests_case(sys.modules[__name__])
        2. run_all_tests_case(your_test_module)
    """
    from console_color import cprint, RGB, create_print, Style  # pip install console_color

    cur_module = module
    print(f'working on {cprint(cur_module.__file__, RGB.GREEN, pf=False)}')
    bp = create_print(fore=RGB.BLUE, bg=RGB.YELLOW, style=Style.ITALIC, pf=False)

    def blue_print(msg):
        return bp(' ' + str(msg) + ' ')

    tests_class_list = [the_class for class_name, the_class in inspect.getmembers(cur_module, inspect.isclass) if class_name.endswith('Tests')]
    suite_list = [unittest.TestLoader().loadTestsFromTestCase(class_module) for class_module in tests_class_list]
    suite_class_set = unittest.TestSuite(suite_list)

    # suite_function_set = unittest.TestSuite()
    # suite_function_set.addTest(CLITests('test_show_version'))

    suite = suite_class_set  # pick one of two: suite_class_set, suite_function_set
    result = unittest.TextTestRunner(verbosity=1).run(suite)  # self.verbosity = 0  # 0, 1, 2.  unittest.TextTestResult
    print('\n'.join([
        f'Run {blue_print(str(result.testsRun))} tests',
        f'errors: {blue_print(str(len(result.errors)))}',
        f'failures: {blue_print(str(len(result.failures)))}']))
    return result


if __name__ == '__main__':
    run_all_tests_case(sys.modules[__name__])
