from typing import Union, Tuple, List, Dict, Optional, Callable
JSON = List[Dict[str, object]]
INPUT_SHAPE = Optional[List[Tuple[int]]]

import numpy
from tennis.backend.api import *

from .exporter import RKNNExporter, RKNNConfig, Calibrator
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
        elif op["op"] == "letterbox":
            size = list(op["size"])
            outer_value = 0 if "outer_value" not in op else op["outer_value"]
            image_filter.letterbox(size[0], size[1], outer_value)
        else:
            raise Exception("Not supported processor: {}".format(op))

    return image_filter


register_device = {
    "target": None,
    "device_id": None,
}


class RKNN2(Operator):
    def __init__(self):
        super(RKNN2, self).__init__()
        self.__rknn_handle = None

    def dispose(self):  # type: () -> None
        if self.__rknn_handle is not None:
            self.__rknn_handle.release()

    def init(self, params, context):
        # type: (OperatorParams, OperatorContext) -> None
        """
        :param params:
        :param context:
        :return: None
        """
        self.__rknn_file = params["rknn_file"].str
        self.__format = params["format"].str
        format_map = {
            "NCHW": "nchw",
            "NHWC": "nhwc",
            "nchw": "nchw",
            "nhwc": "nhwc",
        }
        assert self.__format in format_map.keys()
        self.__format = format_map[self.__format]
        if not os.path.isfile(self.__rknn_file):
            raise FileNotFoundError("Can not found: {}".format(self.__rknn_file))

        output_shapes = params["output_shapes"].numpy
        self.__output_shapes = []
        N = output_shapes[0]
        offset = 1
        for i in range(N):
            size = output_shapes[offset]
            offset += 1
            self.__output_shapes.append(list(output_shapes[offset:offset + size]))
            offset += size

        target = register_device["target"] if "target" in register_device else None
        device_id = register_device["device_id"] if "device_id" in register_device else None

        from rknn.api import RKNN
        self.__rknn_handle = RKNN(verbose=False)
        self.__rknn_handle.config(mean_values=None,
                                  std_values=None,
                                  target_platform=target
                                  ) # not set quantized_dtype
        ret = self.__rknn_handle.load_rknn(path=self.__rknn_file)
        if ret != 0:
            raise Exception("RKNN failed with ret={}".format(ret))

        if target is not None and device_id is not None:
            print("[I] RKNN init runtime {} {}".format(target, device_id))
            ret = self.__rknn_handle.init_runtime(target=target, device_id=device_id)
        else:
            print("[I] RKNN init runtime {}".format('simulator'))
            ret = self.__rknn_handle.init_runtime(target=target)
        # ret = self.__rknn_handle.init_runtime(target="rk3399pro", device_id="FB0GBDHQKS")
        # ret = self.__rknn_handle.init_runtime()
        # print(self.__rknn_handle.list_devices())
        if ret != 0:
            raise Exception("RKNN failed with ret={}".format(ret))

    def run(self, args, context):
        # type: (List[Tensor], OperatorContext) -> Union[Tensor, List[Tensor]]
        """
        :param args:
        :param context:
        :return: list of tuple like [(FLOAT32, (1, 3, 4, 4)), (INT32, (1, 2))]
        """
        numpy_args = [args[i].numpy for i in range(len(args))]
        print("[I] RKNN input: {}".format([i.shape for i in numpy_args]))
        import time
        begin = time.time()
        outputs = self.__rknn_handle.inference(inputs=numpy_args, data_format=self.__format)
        end = time.time()
        print("[I] RKNN inference spent: {}s".format(end - begin))
        if len(outputs) != len(self.__output_shapes):
            raise Exception("Output count mismatch, got {} but {} wanted.".format(
                len(outputs), len(self.__output_shapes)))
        outputs = [numpy.reshape(outputs[i], self.__output_shapes[i]) for i in range(len(outputs))]
        print("[I] RKNN output: {}".format([output.shape for output in outputs]))
        return outputs


RegisterOperator(RKNN2, "cpu", "rknn2")


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


def list_image_files(root, path="", result=None):
    image_ext = {".jpg", ".png", ".bmp"}
    if result is None:
        result = []
    root_path = os.path.join(root, path)
    if os.path.isdir(root_path):
        for filename in os.listdir(root_path):
            path_filename = os.path.join(path, filename)
            filepath = os.path.join(root, path_filename)
            if os.path.isdir(filepath):
                list_image_files(root, path_filename, result)
                continue
            _, ext = os.path.splitext(filename)
            if ext not in image_ext:
                continue
            result.append(path_filename)
    else:
        _, ext = os.path.splitext(path)
        if ext not in image_ext:
            return
        result.append(path)
    return result


class SampleCalibrator(Calibrator):
    def __init__(self, dataset, limit=-1):
        # type: (str, int) -> None
        """
        root or image list
        :param dataset:
        """
        if sys.version > '3':
            basestring = str
        image_ext = {".jpg"}
        dataroot = ""
        filelist = []
        if os.path.isdir(dataset):
            dataroot = os.path.abspath(dataset)
            filelist = list_image_files(dataroot)
            filelist.sort()
        elif os.path.isfile(dataset):
            dataroot = os.getcwd()
            with open(dataset, "r") as f:
                for line in f.readlines():
                    assert isinstance(line, basestring)
                    filelist.append(line.strip())
        else:
            raise ValueError("param 1 must be existed path or file")
        self.__filelist = [path if os.path.isabs(path) else os.path.join(dataroot, path)
                           for path in filelist]
        self.__next_index = 0
        self.__image_filter = None

        if 0 < limit < len(self.__filelist):
            import random
            random.seed(4481)
            self.__filelist = random.sample(self.__filelist, limit)

    @property
    def image_filter(self):
        return self.__image_filter

    @image_filter.setter
    def image_filter(self, value):
        assert callable(value)
        self.__image_filter = value

    def number(self):  # type: () -> int
        return len(self.__filelist)

    def next(self):
        # type: () -> Optional[List[numpy.ndarray]]
        while True:
            if self.__next_index >= len(self.__filelist):
                return None
            filepath = self.__filelist[self.__next_index]
            self.__next_index += 1
            image = cv2.imread(filepath)
            if image is None:
                print("[WARNING]: Fail to open: {}".format(filepath))
                continue
            data = numpy.expand_dims(image, 0)

            if self.image_filter is not None:
                data = self.image_filter(data)

            return [data, ]

    def reset(self):
        n = self.__next_index
        self.__next_index = 0
        return n


class SampleImageFilter(object):
    def __init__(self, image_filter, device=None):
        if device is None:
            device = Device("cpu")
        self.__workbench = Workbench(device=device)
        assert callable(image_filter)
        self.__image_filter = image_filter

    def dispose(self):
        self.__workbench.dispose()

    def __call__(self, image):
        self.__workbench.setup_context()
        return self.__image_filter(image)


def test_tsm_with_rknn(tsm,
                       input_shape,
                       tsm_with_rknn,
                       image,
                       image_filter,
                       config,
                       dataset=None,
                       use_buffer=True):
    # type: (str, INPUT_SHAPE, Optional[str], Union[str, numpy.ndarray], Union[None, JSON, Callable], RKNNConfig, Union[str, Calibrator], bool) -> None
    if not os.path.isfile(tsm):
        raise FileNotFoundError(tsm)

    limit = -1
    if isinstance(dataset, (list, tuple)):
        assert len(dataset) == 2
        dataset, limit = dataset[0], dataset[1]

    exporter = RKNNExporter()
    exporter.config = config

    register_device["target"] = config.target
    register_device["device_id"] = config.device_id

    if tsm_with_rknn is None:
        import hashlib
        md5 = "0"
        with open(tsm, 'rb') as fp:
            data = fp.read()
            md5 = hashlib.md5(data).hexdigest()
        tsm_with_rknn = exporter.suggest_name("/tmp/tennis_rknn/" + md5, tsm)

    if os.path.isdir(tsm_with_rknn):
        tsm_with_rknn = exporter.suggest_name(tsm_with_rknn, tsm)

    device = Device("cpu", 0)

    need_dispose_filter = False
    if image_filter is not None and not callable(image_filter):
        image_filter = _load_image_filter(image_filter, device)
        need_dispose_filter = True

    if not use_buffer or not os.path.isfile(tsm_with_rknn):
        calibrator = None
        calibrator_filter = None
        if config.do_quantization:
            assert dataset is not None, "dataset must be set if config.do_quantization is True."
            if isinstance(dataset, Calibrator):
                calibrator = dataset
                # if image_filter is not None:
                #    logger.warning("image_filter will be ignored, as calibrator was given.")
            else:
                calibrator = SampleCalibrator(dataset, limit)
                if image_filter is not None:
                    calibrator_filter = SampleImageFilter(image_filter, device=device)
                    calibrator.image_filter = calibrator_filter
                else:
                    logger.warning("image_filter should be set if given dataset path only.")
        elif dataset is not None:
            # load dataset if given, for sample test
            if isinstance(dataset, Calibrator):
                calibrator = dataset
            else:
                calibrator = SampleCalibrator(dataset, limit)
                if image_filter is not None:
                    calibrator_filter = SampleImageFilter(image_filter, device=device)
                    calibrator.image_filter = calibrator_filter
                else:
                    logger.warning("image_filter should be set if given dataset path only.")

        logger.info("Loading tsm...")
        exporter.load(module=tsm, input_shape=input_shape)
        logger.info("Exporting tsm with rknn...")
        exporter.export_tsm_with_rknn(tsm_with_rknn, calibrator)
        if calibrator_filter is not None:
            calibrator_filter.dispose()

    if isinstance(image, str):
        image = cv2.imread(image)

    logger.info("Extract original module...")
    feature1 = _export_feature(tsm, device, image, image_filter)

    if config.device_id is None:
        logger.error("Test rknn inference not work on simulator for target: {}".format(config.target))
        return

    logger.info("Extract with rknn module...")
    feature2 = _export_feature(tsm_with_rknn, device, image, image_filter)
    if len(feature1) != len(feature2):
        logger.warning("Output count mismatch. {} vs. {}.".format(len(feature1), len(feature2)))

    # print("Original module inference: ", numpy.asarray(feature1).reshape([-1]))
    # print("    RKNN module inference: ", numpy.asarray(feature2).reshape([-1]))

    for a, b in zip(feature1, feature2):
        detail = compare(a, b)
        avg, max = diff(a, b)
        detail = list(numpy.asarray(detail).reshape([-1]))
        cos = compare(a.reshape([-1]), b.reshape([-1]))
        logger.info("Output: {} vs. {}".format(a.shape, b.shape))
        logger.info("Output: avg={}, max={}, cos={}, detail={}".format(avg, max, cos, detail))

    if need_dispose_filter:
        image_filter.dispose()

    logger.info("Test done.")

    import gc
    gc.collect()




