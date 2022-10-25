import sys
import time
import cv2
import numpy as np
tennis = '/home/kier/git/TensorStack/python'
sys.path.append(tennis)

from tennis.backend.api import *
from tennis_rknn import onnx_tools as tools


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
    img_path = "debug.jpg"

    # tools.test_tsm_with_onnx("/home/kier/Documents/rknn-toolkit-v0.9.3/rknn_example/onnx/arcface_112x112/resnet50_iter_118000.tsm",
    #                          None,
    #                          None,
    #                          img_path,
    #                          [
    #                              {"op": "resize", "size": [112, 112]},
    #                              {"op": "to_float"},
    #                              {"op": "sub_mean", "mean": [0.0, 0.0, 0.0]},
    #                              {"op": "div_std", "std": [255.0, 255.0, 255.0]},
    #                              {"op": "to_chw"},
    #                          ])

    # tools.test_tsm_with_onnx("/home/kier/working/model/mobilefacenet-v1_arcface_lffd-plfd_0519.tsm",
    #                          None,
    #                          None,
    #                          img_path,
    #                          [
    #                              {"op": "resize", "size": [112, 112]},
    #                              {"op": "to_float"},
    #                              {"op": "sub_mean", "mean": [0.0, 0.0, 0.0]},
    #                              {"op": "div_std", "std": [255.0, 255.0, 255.0]},
    #                              {"op": "to_chw"},
    #                          ], False)

    root = "/home/kier/working/nnie/rawmd/"
    #
    # tools.test_tsm_with_onnx(root + "_acc905_squeezenet_v15_90_90_closs1_2DB_4class_214000_1010.tsm",
    #                          [1, 102, 102, 3],
    #                          None,
    #                          img_path,
    #                          [
    #                              {"op": "resize", "size": [102, 102]},
    #                              {"op": "to_float"},
    #                          ])
    #
    # tools.test_tsm_with_onnx(root + "emotions_v1.2_2019-10-17.tsm",
    #                          None,
    #                          None,
    #                          img_path,
    #                          [{"op": "resize", "size": [248, 248]}] +
    #                             load_json_pre_processor(root + "emotion_recognizer_v1.json"))

    # tools.test_tsm_with_onnx(root + "emotions_v2_2019-10-17.tsm",
    #                          None,
    #                          None,
    #                          img_path,
    #                          [{"op": "resize", "size": [256, 256]}] +
    #                          load_json_pre_processor(root + "emotion_recognizer_v2.json"))

    # tools.test_tsm_with_onnx(root + "faceboxes_ipc_2019-7-12.tsm",
    #                          [1, 3, 300, 300],
    #                          None,
    #                          img_path,
    #                          [{"op": "resize", "size": [300, 300]}] +
    #                          load_json_pre_processor(root + "faceboxes_ipc_2019-7-12.json"))
    #
    # tools.test_tsm_with_onnx(root + "lffd_v1_bn.tsm",
    #                          [1, 3, 300, 300],
    #                          None,
    #                          img_path,
    #                          [
    #                              {"op": "resize", "size": [300, 300]},
    #                              {"op": "to_float"},
    #                              {"op": "sub_mean", "mean": [127.5, 127.5, 127.5]},
    #                              {"op": "div_std", "std": [127.5, 127.5, 127.5]},
    #                              {"op": "to_chw"},
    #                          ], False)


    # root = "/home/kier/working/bi/model/"
    # dataset = "/home/kier/working/bi/data/fas_global/"
    # testcase = dataset + "4090.jpg"
    # tools.test_tsm_with_onnx(root + "SeetaAntiSpoofing.plg.1.0.m01d29.tsm",
    #                    [1, 3, 300, 300],
    #                    None,
    #                    testcase,
    #                    [
    #                        {"op": "to_float"},
    #                        {"op": "scale", "scale": 0.00784313771874},
    #                        {"op": "sub_mean", "mean": [1.0]},
    #                        {"op": "to_chw"},
    #                    ],
    #                    False)

    # root = "/home/kier/working/bi/model/"
    # dataset = "/home/kier/working/bi/data/fas_local/"
    # testcase = dataset + "download/spoof/0208/up/xl_18/000000.png"
    # fas_image_filter = FASImageFilter()
    # tools. test_tsm_with_onnx(root + "_acc982_res18_longmao1th_1022_11000.tsm",
    #                    [1, 3, 224, 224],
    #                    None,
    #                    testcase,
    #                    fas_image_filter,
    #                    False)

    root = "/home/kier/working/bi/model/"
    dataset = "/home/kier/working/bi/data/fas_nir/"
    testcase = dataset + "spoof/100_9.996234439313412E-5_nir.png"
    nir_image_filter = NIRImageFilter()
    tools.test_tsm_with_onnx(root + "nirFaceAntiSpoofingGrayV1.0.4.tsm",
                       [1, 1, 248, 248],
                       None,
                       testcase,
                       nir_image_filter,
                       False)


if __name__ == '__main__':
    test()
