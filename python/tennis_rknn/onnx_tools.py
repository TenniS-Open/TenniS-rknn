from typing import Union, Tuple, List, Dict, Optional, Callable
JSON = List[Dict[str, object]]
INPUT_SHAPE = Optional[List[Tuple[int]]]

import numpy
from tennis.backend.api import *

from .exporter import RKNNExporter
import os
import cv2

from . import logger


def _load_workbench(path, device=None):
    # type: (str, Optional[Device]) -> Workbench
    if device is None:
        device = Device("cpu", 0)

    module = Module.Load(path)
    bench = Workbench.Load(module=module, device=device)
    bench.set_computing_thread_number(8)
    module.dispose()

    return bench


def _export_feature(path, device, image, image_filter):
    # type: (str, Device, numpy.ndarray, Optional[ImageFilter]) -> List[numpy.ndarray]
    root, _ = os.path.split(path)
    cwd = os.getcwd()
    os.chdir(root)

    bench = _load_workbench(path, device)
    bench.setup_context()

    assert len(image.shape) == 3
    image = numpy.expand_dims(image, 0)
    if image_filter is not None:
        image = image_filter(image)

    bench.input(0, image)
    bench.run()

    outputs = []
    for i in range(bench.output_count()):
        output = bench.output(i)
        outputs.append(output.numpy)
        output.dispose()

    bench.dispose()

    os.chdir(cwd)

    return outputs


def _load_image_filter(obj, device):
    # type: (JSON, Device) -> ImageFilter
    image_filter = ImageFilter(device=device)

    for op in obj:
        if "op" not in op:
            raise Exception("Not supported processor: {}".format(op))
        elif op["op"] == "force_gray":
            image_filter.force_gray()
        elif op["op"] == "resize":
            size = op["size"]
            image_filter.resize(width=size[0], height=size[1])
        elif op["op"] == "to_float":
            image_filter.to_float()
        elif op["op"] == "to_chw":
            image_filter.to_chw()
        elif op["op"] == "norm_image":
            image_filter.norm_image(float(op["epsilon"]))
        elif op["op"] == "sub_mean":
            image_filter.sub_mean(op["mean"])
        elif op["op"] == "div_std":
            image_filter.div_std(op["std"])
        elif op["op"] == "center_crop":
            size = op["size"]
            image_filter.center_crop(width=size[0], height=size[1])
        elif op["op"] == "channel_swap":
            image_filter.channel_swap(op["shuffle"])
        elif op["op"] == "scale":
            image_filter.scale(op["scale"])
        else:
            raise Exception("Not supported processor: {}".format(op))

    return image_filter


class ONNX(Operator):
    def __init__(self):
        super(ONNX, self).__init__()
        self.__onnx_file = None
        #self.__session = None

    def dispose(self):  # type: () -> None
        pass

    def init(self, params, context):
        # type: (OperatorParams, OperatorContext) -> None
        """
        :param params:
        :param context:
        :return: None
        """
        self.__onnx_file = params["onnx_file"].str
        import onnxruntime
        self.__session = onnxruntime.InferenceSession(self.__onnx_file)
        self.__inputs = self.__session.get_inputs()
        self.__outputs = self.__session.get_outputs()
        self.__input_names = [node.name for node in self.__inputs]
        self.__output_names = [node.name for node in self.__outputs]

    def run(self, args, context):
        # type: (List[Tensor], OperatorContext) -> Union[Tensor, List[Tensor]]
        """
        :param args:
        :param context:
        :return: list of tuple like [(FLOAT32, (1, 3, 4, 4)), (INT32, (1, 2))]
        """
        if len(args) != len(self.__input_names):
            raise Exception("Must input {} tensor, got {}.".format(len(self.__input_names), len(args)))
        outputs = self.__session.run(self.__output_names, {self.__input_names[i]: args[i].numpy for i in range(len(args))})
        return outputs


RegisterOperator(ONNX, "cpu", "onnx")


def compare(e1, e2, epsilon=1e-5):
    np = numpy
    _y1 = np.sqrt(np.sum(np.square(e1), axis=-1))
    _y2 = np.sqrt(np.sum(np.square(e2), axis=-1))
    y1_y2 = np.sum(np.multiply(e1, e2), axis=-1)
    cosine = y1_y2 / (_y1 * _y2 + epsilon)
    return cosine


def diff(e1, e2):
    np = numpy
    _y1 = np.reshape(e1, [-1])
    _y2 = np.reshape(e2, [-1])
    size = min(len(_y1), len(_y2))

    sum = 0
    max = 0

    for i in range(size):
        diff = abs(_y1[i] - _y2[i])
        sum += diff
        if diff > max:
            max = diff

    avg = sum / size

    return avg, max


def test_tsm_with_onnx(tsm,
                       input_shape,
                       tsm_with_onnx,
                       image,
                       image_filter,
                       use_buffer=True):
    # type: (str, INPUT_SHAPE, Optional[str], Union[str, numpy.ndarray], Union[None, JSON, Callable], bool) -> None
    if not os.path.isfile(tsm):
        raise FileNotFoundError(tsm)
    if tsm_with_onnx is None:
        import hashlib
        md5 = "0"
        with open(tsm, 'rb') as fp:
            data = fp.read()
            md5 = hashlib.md5(data).hexdigest()
        path, name_ext = os.path.split(tsm)
        name, ext = os.path.splitext(name_ext)
        tsm_with_onnx = "/tmp/tennis_onnx/{}.onnx.{}{}".format(name, md5, ext)

    if os.path.isdir(tsm_with_onnx):
        path, name_ext = os.path.split(tsm)
        name, ext = os.path.splitext(name_ext)
        tsm_with_onnx = os.path.join(tsm_with_onnx, "{}.onnx{}".format(name, ext))

    if not use_buffer or not os.path.isfile(tsm_with_onnx):
        exporter = RKNNExporter()
        exporter.load(module=tsm, input_shape=input_shape)
        exporter.export_onnx(tsm_with_onnx, subdir="onnx", export_main=True)

    device = Device("cpu", 0)
    if isinstance(image, str):
        image = cv2.imread(image)

    need_dispose_filter = False
    if image_filter is not None and not callable(image_filter):
        image_filter = _load_image_filter(image_filter, device)
        need_dispose_filter = True

    logger.info("Extract original module...")
    feature1 = _export_feature(tsm, device, image, image_filter)

    logger.info("Extract with onnx module...")
    feature2 = _export_feature(tsm_with_onnx, device, image, image_filter)
    if len(feature1) != len(feature2):
        logger.warning("Output count mismatch. {} vs. {}.".format(len(feature1), len(feature2)))

    # print("Original module inference: ", numpy.asarray(feature1).reshape([-1]))
    # print("    ONNX module inference: ", numpy.asarray(feature2).reshape([-1]))

    for a, b in zip(feature1, feature2):
        detail = compare(a, b)
        avg, max = diff(a, b)
        detail = list(numpy.asarray(detail).reshape([-1]))
        cos = compare(a.reshape([-1]), b.reshape([-1]))
        logger.info("Output: avg={}, max={}, cos={}, detail={}".format(avg, max, cos, detail))

    if need_dispose_filter:
        image_filter.dispose()

    logger.info("Test done.")

    import gc
    gc.collect()


