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
    config.device_id = "TUXG0IBTRB"
    config.do_quantization = True
    config.quantized_dtype = RKNNConfig.asymmetric_quantized_u8
    # config.quantized_dtype = RKNNConfig.dynamic_fixed_point_8
    # config.quantized_dtype = RKNNConfig.dynamic_fixed_point_16
    limit = 2000
    use_buffer = True
    output_dir = "/home/kier/working/bi/rknn/"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


    # dataset = "/home/kier/Documents/rknn-toolkit-v0.9.3/rknn_example/onnx/arcface_112x112/data/fr_crop_112x112_2k/"
    # img_path = "/home/kier/Documents/rknn-toolkit-v0.9.3/rknn_example/onnx/arcface_112x112/data/fr_crop_112x112_2k/00000.jpg"
    # test_tsm_with_rknn("/home/kier/Documents/rknn-toolkit-v0.9.3/rknn_example/onnx/arcface_112x112/resnet50_iter_118000.tsm",
    #                          None,
    #                          None,
    #                          img_path,
    #                          [
    #                              {"op": "resize", "size": [112, 112]},
    #                              {"op": "to_float"},
    #                              {"op": "sub_mean", "mean": [0.0, 0.0, 0.0]},
    #                              {"op": "div_std", "std": [255.0, 255.0, 255.0]},
    #                              {"op": "to_chw"},
    #                          ],
    #                          config,
    #                          dataset,
    #                          False)

    root = "/home/kier/working/bi/model/"

    dataset = "/home/kier/working/bi/data/fd_to_320x320/images/"
    testcase = dataset + "01262.jpg"
    test_tsm_with_rknn(root + "lffd_v2_wwm_nobn_bgs_1.tsm",
                       [1, 3, 320, 320],
                       output_dir,
                       testcase,
                       [
                           {"op": "resize", "size": [320, 320]},
                           {"op": "to_float"},
                           {"op": "sub_mean", "mean": [127.5, 127.5, 127.5]},
                           {"op": "div_std", "std": [127.5, 127.5, 127.5]},
                           {"op": "to_chw"},
                       ],
                       config,
                       [dataset, limit],
                       use_buffer)

    dataset = "/home/kier/working/bi/data/fr_quantization_248x248/"
    testcase = dataset + "0002720_1984.jpg"
    test_tsm_with_rknn(root + "RN30.light.tsm",
                       [1, 3, 248, 248],
                       output_dir,
                       testcase,
                       [
                           {"op": "center_crop", "size": [248, 248]},
                           {"op": "to_float"},
                           {"op": "to_chw"},
                       ],
                       config,
                       [dataset, limit],
                       use_buffer)

    dataset = "/home/kier/working/bi/data/fr_quantization_248x248/"
    testcase = dataset + "0002720_1984.jpg"
    test_tsm_with_rknn(root + "RN50.D.K.A.tsm",
                       [1, 3, 248, 248],
                       output_dir,
                       testcase,
                       [
                           {"op": "center_crop", "size": [248, 248]},
                           {"op": "to_float"},
                           {"op": "to_chw"},
                       ],
                       config,
                       [dataset, limit],
                       use_buffer)

    dataset = "/home/kier/working/bi/data/PoseImages/"
    testcase = dataset + "1576636621773841.jpg"
    test_tsm_with_rknn(root + "PoseEstimation1.1.0.tsm",
                       [1, 3, 90, 90],
                       output_dir,
                       testcase,
                       [
                           {"op": "to_float"},
                           {"op": "to_chw"},
                       ],
                       config,
                       [dataset, limit],
                       use_buffer)

    dataset = "/home/kier/working/bi/data/PoseImages/"
    testcase = dataset + "1576636621773841.jpg"
    test_tsm_with_rknn(root + "PoseEstimation1.1.0.tsm",
                       [1, 3, 90, 90],
                       output_dir,
                       testcase,
                       [
                           {"op": "to_float"},
                           {"op": "to_chw"},
                       ],
                       config,
                       [dataset, limit],
                       use_buffer)

    dataset = "/home/kier/working/bi/data/fas_global/"
    testcase = dataset + "4090.jpg"
    test_tsm_with_rknn(root + "SeetaAntiSpoofing.plg.1.0.m01d29.tsm",
                       [1, 3, 300, 300],
                       output_dir,
                       testcase,
                       [
                           {"op": "to_float"},
                           {"op": "scale", "scale": 0.00784313771874},
                           {"op": "sub_mean", "mean": [1.0]},
                           {"op": "to_chw"},
                       ],
                       config,
                       [dataset, limit],
                       use_buffer)

    dataset = "/home/kier/working/bi/data/fas_local/"
    testcase = dataset + "download/spoof/0208/up/xl_18/000000.png"
    fas_calibrator = SampleCalibrator(dataset, limit)
    fas_image_filter = FASImageFilter()
    fas_calibrator.image_filter = fas_image_filter
    test_tsm_with_rknn(root + "_acc982_res18_longmao1th_1022_11000.tsm",
                       [1, 3, 224, 224],
                       output_dir,
                       testcase,
                       fas_image_filter,
                       config,
                       fas_calibrator,
                       use_buffer)

    dataset = "/home/kier/working/bi/data/LBN/"
    testcase = dataset + "blur_20191203_testDb_49.jpg"
    test_tsm_with_rknn(root + "_loss034018_squeezenetV22_blur1_2th_relabel_18000_1209.tsm",
                       [1, 256, 256, 3],
                       output_dir,
                       testcase,
                       [
                           {"op": "resize", "size": [256, 256]},
                           {"op": "to_float"},
                       ],
                       config,
                       [dataset, limit],
                       use_buffer)

    nir_config = copy.copy(config)
    nir_config.channel_mean_value = None
    nir_config.reorder_channel = None
    dataset = "/home/kier/working/bi/data/fas_nir/"
    testcase = dataset + "spoof/100_9.996234439313412E-5_nir.png"
    nir_calibrator = SampleCalibrator(dataset, limit)
    nir_image_filter = NIRImageFilter()
    nir_calibrator.image_filter = nir_image_filter
    test_tsm_with_rknn(root + "nirFaceAntiSpoofingGrayV1.0.4.tsm",
                       [1, 1, 248, 248],
                       output_dir,
                       testcase,
                       nir_image_filter,
                       nir_config,
                       nir_calibrator,
                       use_buffer)


if __name__ == '__main__':
    test()
