import sys
import time
import cv2
import numpy as np
tennis = '/home/kier/git/TensorStack/python'
sys.path.append(tennis)

from tennis.backend.api import *
from tennis_rknn.rknn_tools import test_tsm_with_rknn, RKNNConfig, SampleCalibrator

import os
import numpy
import cv2
import copy


def load_json_pre_processor(path):
    with open(path, "r") as f:
        import json
        obj = json.load(f)
        assert "pre_processor" in obj
        return obj["pre_processor"]


class FASImageFilter(object):
    def __init__(self):
        device = Device("cpu")
        self.__workbench = Workbench(device=device)
        self.__image_filter = ImageFilter(device=device)
        self.__image_filter.center_crop(224, 224)
        self.__image_filter.to_float()
        self.__image_filter.to_chw()

    def dispose(self):
        self.__image_filter.dispose()
        self.__workbench.dispose()

    def __call__(self, image):
        self.__workbench.setup_context()

        image = image[0]
        # do image actions
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        image = numpy.expand_dims(image, 0)

        input = Tensor(image)
        output = self.__image_filter.run(input)
        output_numpy = output.numpy
        input.dispose()
        output.dispose()
        return output_numpy


class NIRImageFilter(object):
    def __init__(self):
        device = Device("cpu")
        self.__workbench = Workbench(device=device)
        self.__image_filter = ImageFilter(device=device)
        self.__image_filter.to_float()
        self.__image_filter.to_chw()

    def dispose(self):
        self.__image_filter.dispose()
        self.__workbench.dispose()

    def __call__(self, image):
        self.__workbench.setup_context()

        image = image[0]
        # do image actions

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)
        image = image[:, :, numpy.newaxis]

        image = numpy.expand_dims(image, 0)

        input = Tensor(image)
        output = self.__image_filter.run(input)
        output_numpy = output.numpy
        input.dispose()
        output.dispose()
        return output_numpy


def test():
    config = RKNNConfig(None, None)
    config.target = "rk3399pro"
    config.device_id = "07SS4Z8H30"
    config.do_quantization = True
    config.quantized_dtype = RKNNConfig.asymmetric_quantized_u8
    # config.quantized_dtype = RKNNConfig.dynamic_fixed_point_8
    # config.quantized_dtype = RK`NNConfig.dynamic_fixed_point_16
    config.pre_compile = False
    config.verbose = False
    limit = 2000
    use_buffer = True
    output_dir = "/home/kier/model/dish/rknn/"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    root = "/home/kier/model/dish/"
    testcase = root + "data/1.png"

    try:
        dataset = root + "data/quantify_dataset/3CLASS/3class/"
        test_tsm_with_rknn(root + "dish_detector_3class_20210201.tsm",
                           [1, 3, 480, 480],
                           output_dir,
                           testcase,
                           [
                               {"op": "letterbox", "size": [480, 480]},
                               {"op": "channel_swap", "shuffle": [2, 1, 0]},
                               {"op": "to_float"},
                               {"op": "scale", "scale": 1 / 255.0},
                               {"op": "to_chw"},
                           ],
                           config,
                           [dataset, limit],
                           use_buffer)
    except Exception as e:
        sys.stderr.write("{}\n".format(e))

    exit()

    try:
        dataset = root + "data/quantify_dataset/6CLASS/6class/"
        test_tsm_with_rknn(root + "dish_detector_6class_20210201.tsm",
                           [1, 3, 480, 480],
                           output_dir,
                           testcase,
                           [
                               {"op": "letterbox", "size": [480, 480]},
                               {"op": "channel_swap", "shuffle": [2, 1, 0]},
                               {"op": "to_float"},
                               {"op": "scale", "scale": 1 / 255.0},
                               {"op": "to_chw"},
                           ],
                           config,
                           [dataset, limit],
                           use_buffer)
    except Exception as e:
        sys.stderr.write("{}\n".format(e))


    # try:
    #     dataset = root + "data/recognizer/data"
    #     testcase = dataset + "/a5_braisedchicken_green_0000.jpg"
    #     test_tsm_with_rknn(root + "dish_recognition.tsm",
    #                        [1, 3, 224, 224],
    #                        output_dir,
    #                        testcase,
    #                        [
    #                            {"op": "resize", "size": [224, 224]},
    #                            {"op": "channel_swap", "shuffle": [2, 1, 0]},
    #                            {"op": "to_float"},
    #                            {"op": "to_chw"},
    #                        ],
    #                        config,
    #                        [dataset, limit],
    #                        use_buffer)
    # except Exception as e:
    #     sys.stderr.write("{}\n".format(e))


if __name__ == '__main__':
    test()
