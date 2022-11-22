#!/usr/bin/env python

from typing import Union
import tennis as ts

from typing import Tuple, List, Dict, Optional, Iterable
import numpy
import os

from tennisfence.spliter import MainGraph
from tennisbuilder.export import fridge
from . import onnx_spliter
from . import onnx_fence

from . import onnx_graph
from tennisbuilder.export import dumper
import copy

import sys
if sys.version > "3":
    basestring = str

"""
For add new converter, See .onnx_spliter.get_spliter for graph spliter; .onnx_caffe for graph converter
"""


def split_onnx(input_tsm, output_tsm, subdir=None, input_shape=None, export_main=False):
    # type: (Union[str, ts.Module], str, str, Union[List[Tuple[int]], Dict[str, Tuple[int]]], bool) -> [None, MainGraph]
    """
    Split support node to sub graph.
    :param input_tsm:
    :param output_tsm:
    :param subdir:
    :return:
    Notice: output main tsm module and sub onnx models
    Every output onnx named {output_tsm}.<i>.onnx
    """
    assert isinstance(subdir, (type(None), basestring))
    module = input_tsm
    if isinstance(module, basestring):
        with open(module, "rb") as f:
            module = ts.Module.Load(f)
    assert isinstance(module, ts.Module)
    filepath = os.path.abspath(output_tsm)
    output_root, filename_ext = os.path.split(filepath)
    filename, ext = os.path.splitext(filename_ext)
    if filename[0] == '.':
        filename = filename[1:]

    output_onnx_root = output_root
    if subdir is not None:
        output_onnx_root = os.path.join(output_root, subdir)

    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    if not os.path.isdir(output_onnx_root):
        os.makedirs(output_onnx_root)

    outputs = module.outputs
    inputs = module.inputs
    print("[INFO]: Freezing graph...")
    outputs, inputs = fridge.freeze(outputs, inputs, input_shape)
    print("[INFO]: Split graph...")
    outputs, inputs = onnx_fence.get_fence().convert(outputs, after=inputs)
    main_graph = onnx_spliter.get_spliter().split(outputs, inputs)
    print("[INFO]: Convert graph...")
    sub_graph_count = main_graph.sub_count()
    for i in range(sub_graph_count):
        output_name_body = "{}.{}".format(filename, i)
        print("[INFO]: Exporting... {}.onnx".format(
            os.path.relpath(os.path.join(output_onnx_root, output_name_body), output_root)))
        output_onnx = "{}.onnx".format(output_name_body)
        sub_node = main_graph.sub_node(i)
        sub_graph = main_graph.sub_graph(i)
        onnx_graph.convert(sub_graph.outputs, sub_graph.inputs,
                           os.path.join(output_onnx_root, output_onnx),
                           version=9)

    if export_main:
        print("[INFO]: Exporting... {}".format(filepath))
        main_module = ts.Module()
        main_module.load(main_graph.outputs)
        main_module.sort_inputs(main_graph.inputs)

        with open(filepath, "wb") as f:
            ts.Module.Save(f, main_module)

    return main_graph


def export_image_list(module, output_names, calibrator, main, output_root, cache=None, device="cpu", device_id=0):
    # type: (ts.Module, List[List[str]], dumper.Calibrator, str, str, str, str, int) -> List[str]
    output_root = os.path.join(output_root, main)

    output_root = os.path.abspath(output_root)
    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    map_unique_output_names = {}
    for name_list in output_names:
        for name in name_list:
            fixed_name = name.replace("/", "=")
            fixed_name = fixed_name.replace("\\", "=")
            map_unique_output_names[name] = fixed_name
    unique_output_names = list(map_unique_output_names.keys())
    map_output_name_index = {}
    for i, name in enumerate(unique_output_names):
        map_output_name_index[name] = i
    list_output_name_path = [os.path.join("npy", "{}".format(i)) for i in range(len(unique_output_names))]
    list_feature_npy = [[], ] * len(unique_output_names)

    # build show data
    if calibrator.number() == 0:
        raise Exception("calibrator.number() must great than 0")

    P = [0, calibrator.number()]

    def process_show():
        sys.stdout.write("\r[{}/{}]   ".format(P[0], P[1]))
        sys.stdout.flush()

    # extract feature
    extractor = dumper.Dumper(module, unique_output_names, calibrator, 1, cache=cache, device=device, device_id=device_id)

    for filepath in list_output_name_path:
        fullpath = os.path.join(output_root, filepath)
        if not os.path.isdir(fullpath):
            os.makedirs(fullpath)

    process_show()

    procceed = 0
    N = 100
    while True:
        features_list = extractor.next()
        if features_list is None:
            break
        for i, name in enumerate(unique_output_names):
            feature_npy = "{}/{:05d}.npy".format(list_output_name_path[i], procceed)
            feature_data = features_list[i]
            feature_data = feature_data.transpose([0, 2, 3, 1])  # save as NHWC format
            numpy.save(os.path.join(output_root, feature_npy), feature_data)
            list_feature_npy[i].append(feature_npy)

        procceed += 1

        P[0] += 1
        process_show()

    process_show()
    print("\n[INFO]: Extract image features done.")

    # write filelist
    dataset_list = []
    for i, name_list in enumerate(output_names):
        sub_graph_dataset_filename = \
            os.path.join(output_root,
                         "{}_[{}].txt".format(i, ",".join([map_unique_output_names[name] for name in name_list])))

        index_list = [map_output_name_index[name] for name in name_list]
        block = '\n'.join([' '.join(npy_files) for npy_files in zip(*[list_feature_npy[i] for i in index_list])])

        with open(sub_graph_dataset_filename, "w") as f:
            f.write(block)
            f.write("\n")

        dataset_list.append(sub_graph_dataset_filename)
    print("[INFO]: Build dataset file list done.")

    return [os.path.join(output_root, path) for path in dataset_list]


class Calibrator(object):
    """
    Return valid set
    """
    def next(self):
        # type: () -> Tuple[numpy.ndarray]
        """
        Get next sample for quantification, tuple means multi inputs
        :return:
        """
        raise NotImplementedError

    def reset():
        # type: () -> int
        """
        Reset simple, return non-zero (sample number) to succeed
        :return:
        """
        return 0


class NetInferer(object):
    def run(self, inputs, outputs):
        # type: (List[numpy.ndarray], List[str]) -> List[numpy.ndarray]
        """
        :param inputs: length input count
        :param outputs: get output names
        :return:
        """
        raise NotImplementedError


class RKNNConfig(object):
    """
    This object is to tell rknn toolkit how to compile onnx module
    """
    def __init__(self, onnx_file, rknn_file):
        # type: (Optional[str], Optional[str]) -> None
        self.__target = None
        self.__device_id = None
        self.__channel_mean_value = None  # [0, 0, 0, 1]
        self.__reorder_channel = None  # default is [2, 1, 0], may change it if channel number is not 3 channels
        self.__dataset = ""
        self.__pre_compile = False
        self.__quantized_dtype = self.asymmetric_quantized_u8
        self.__do_quantization = False
        self.__onnx_file = onnx_file
        self.__rknn_file = rknn_file
        self.__verbose = False

        self.__inputs = []  # input names of origin module
        self.__outputs = []  # input names of origin module

    asymmetric_quantized_8 = "asymmetric_quantized-8"
    asymmetric_quantized_16 = "asymmetric_quantized-16"
    asymmetric_quantized_u8 = "asymmetric_quantized-u8"
    dynamic_fixed_point_8 = "dynamic_fixed_point-8"
    dynamic_fixed_point_16 = "dynamic_fixed_point-16"

    rk3399pro = "rk3399pro"
    rk1808 = "rk1808"
    rk3588 = "rk3588"

    def tag(self):
        tags = list()
        tags.append(self.target if self.target is not None else "simulator")
        if self.__channel_mean_value is None:
            tags.append("[]")
        else:
            tags.append("[{}]".format(",".join([str(i) for i in self.__channel_mean_value])))
        if self.__reorder_channel is None:
            tags.append("[]")
        else:
            tags.append("[{}]".format(",".join([str(i) for i in self.__reorder_channel])))
        tags.append("precompile" if self.pre_compile else "no-compile")
        tags.append("do-quantization" if self.do_quantization else "no-quantization")
        tags.append(self.quantized_dtype if self.do_quantization else "float16")
        return ".".join(tags)

    @property
    def quantized_dtype(self):
        # type: () -> str
        return self.__quantized_dtype

    @quantized_dtype.setter
    def quantized_dtype(self, val):
        allowed_quantized_dtype = {
            "asymmetric_quantized-u8", 'dynamic_fixed_point-8', 'dynamic_fixed_point-16',
            "asymmetric_quantized-8", "asymmetric_quantized-16", 
        }
        assert val in allowed_quantized_dtype
        self.__quantized_dtype = val

    @property
    def channel_mean_value(self):
        # type: () -> Optional[str]
        if self.__channel_mean_value is None:
            return None
        return ' '.join([str(i) for i in self.__channel_mean_value])

    @channel_mean_value.setter
    def channel_mean_value(self, val):
        if val is None:
            self.__channel_mean_value = None
            return
        assert len(val) > 1
        self.__channel_mean_value = val
        # raise Exception("channel_mean_value does not support setting outside")

    @property
    def reorder_channel(self):
        # type: () -> Optional[str]
        if self.__reorder_channel is None:
            return None
        return ' '.join([str(i) for i in self.__reorder_channel])

    @reorder_channel.setter
    def reorder_channel(self, val):
        if val is None:
            self.__reorder_channel = None
            return
        assert len(val) > 0
        self.__reorder_channel = val
        # raise Exception("channel_mean_value does not support setting outside")

    @property
    def target(self):
        return self.__target

    @target.setter
    def target(self, val):
        assert val is None or val in {"rk3588"}
        self.__target = val

    @property
    def device_id(self):
        return self.__device_id

    @device_id.setter
    def device_id(self, val):
        assert isinstance(val, str)
        self.__device_id = val

    @property
    def onnx_file(self):
        return self.__onnx_file

    @onnx_file.setter
    def onnx_file(self, val):
        assert isinstance(val, str)
        self.__onnx_file = val

    @property
    def rknn_file(self):
        return self.__rknn_file

    @rknn_file.setter
    def rknn_file(self, val):
        assert isinstance(val, str)
        self.__rknn_file = val

    @property
    def do_quantization(self):
        return self.__do_quantization

    @do_quantization.setter
    def do_quantization(self, val):
        assert isinstance(val, bool)
        self.__do_quantization = val

    @property
    def dataset(self):
        return self.__dataset

    @dataset.setter
    def dataset(self, val):
        assert isinstance(val, str)
        self.__dataset = val

    @property
    def pre_compile(self):
        return self.__pre_compile

    @pre_compile.setter
    def pre_compile(self, val):
        assert isinstance(val, bool)
        self.__pre_compile = val

    def default_channels(self, c):
        # type: (int) -> None
        if c in {1, 2, 3, 4}:
            self.__channel_mean_value = [0] * c + [1]
        if c in {3}:
            self.__reorder_channel = list(range(c))

    @property
    def verbose(self):
        return self.__verbose

    @verbose.setter
    def verbose(self, val):
        assert isinstance(val, bool)
        self.__verbose = val

    @property
    def inputs(self):
        return self.__inputs

    @inputs.setter
    def inputs(self, val):
        self.__inputs = [str(s) for s in val]

    @property
    def outputs(self):
        return self.__outputs

    @outputs.setter
    def outputs(self, val):
        self.__outputs = [str(s) for s in val]


def _check_input_shape_dict_str_int_list(shape):
    # type: (Dict[str, Tuple[int]]) -> bool
    if not isinstance(shape, dict):
        return False
    for k, v in shape.items():
        if not isinstance(k, str):
            return False
        if not _check_input_shape_int_list(v):
            return False
    return True


def _check_input_shape_int_list(shape):
    # type: (Union[List[int], Tuple[int]]) -> bool
    if not isinstance(shape, (list, tuple)):
        return False
    for i in shape:
        if not isinstance(i, int):
            return False
    return True


def _check_input_shape_list_of_int_list(shape):
    # type: ( List[Tuple[int]]) -> bool
    if not isinstance(shape, (list, tuple)):
        for i in shape:
            if not _check_input_shape_int_list(i):
                return False
    return True


def _check_input_shape(shape):
    # type: (Union[List[int], List[Tuple[int]], Dict[str, Tuple[int]]]) -> Union[List[Iterable[int]], Dict]
    def _error():
        raise Exception("Input shape must be List[int], List[Tuple[int]] or Dict[str, Tuple[int]]")

    if isinstance(shape, dict):
        if not _check_input_shape_dict_str_int_list(shape):
            _error()
        return shape

    if _check_input_shape_int_list(shape):
        return [shape]

    if not _check_input_shape_list_of_int_list(shape):
        _error()

    return shape


class RKNNExporter(object):
    def __init__(self, host_device="cpu", host_device_id=0):
        self.__original_module = None   # update by load
        self.__input_shape = None       # update by load

        self.__cache = None # cache temp files
        self.__host_device = host_device
        self.__host_device_id = host_device_id

        self.__max_batch_size = 1   # default max batch size

        self.__config = RKNNConfig(None, None)
        pass

    @property
    def max_batch_size(self):
        return self.__max_batch_size

    @max_batch_size.setter
    def max_batch_size(self, value):
        value = int(value)
        if not 1 <= value <= 256:
            raise ValueError("max_batch_size must be in [1, 256]")
        self.__max_batch_size = value

    @property
    def config(self):
        # type: () -> RKNNConfig
        return self.__config

    @config.setter
    def config(self, val):
        assert isinstance(val, RKNNConfig)
        self.__config = val

    def load(self, module, input_shape=None):
        # type: (Union[str, ts.Module], Union[List[Tuple[int]], Dict[str, Tuple[int]]]) -> None
        if isinstance(module, basestring):
            print("[INFO]: Loading... {}".format(module))
            with open(module, "rb") as f:
                module = ts.Module.Load(f)
        assert isinstance(module, ts.Module)
        self.__original_module = module
        self.__input_shape = input_shape
        # check input shape must be valid
        if input_shape is not None:
            input_shape = _check_input_shape(input_shape)
            if isinstance(input_shape, (list, tuple)):
                for shape in input_shape:
                    for dim in shape[1:]:
                        if dim <= 0:
                            raise ValueError("Input shape must be definite, got {}".format(input_shape))
            elif isinstance(input_shape, dict):
                for shape in input_shape.values():
                    for dim in shape[1:]:
                        if dim <= 0:
                            raise ValueError("Input shape must be definite, got {}".format(input_shape))

    def export_onnx(self, filename, subdir=None, export_main=False):
        # type: (str, str, bool) -> None
        if self.__original_module is None:
            raise ValueError("call load fist be before export_onnx")

        output_root, output_name, output_ext = self._split_root_name_ext(filename)
        # 1. split caffe
        main_graph = split_onnx(self.__original_module, filename, subdir, self.__input_shape, export_main=False)
        if not export_main:
            return

        # 2. get image list
        sub_graph_inputs = set()
        sub_graph_count = main_graph.sub_count()
        for i in range(sub_graph_count):
            for input in main_graph.sub_graph(i).inputs:
                sub_graph_inputs.add(input.name)
        sub_graph_inputs = list(sub_graph_inputs)

        summery_configs = []
        # 3. write nnie cfg file
        for i in range(sub_graph_count):
            node = main_graph.sub_node(i)
            graph = main_graph.sub_graph(i)

            # ref wk filename
            # rknn_instruction_name = os.path.join("rknn", "{}.{}".format(output_name, i))
            # rknn_filename = rknn_instruction_name + ".rknn"
            # print("[INFO]: Waiting... {}".format(rknn_filename))

            onnx_filename = os.path.join(subdir, "{}.{}.onnx".format(output_name, i))

            # update node
            node.op = "onnx"
            node.set("input_count", len(graph.inputs), numpy.int32)     # required
            node.set("output_count", len(graph.outputs), numpy.int32)   # required
            node.set("onnx_file", onnx_filename)    # required

        # 4. write main tsm file
        main_module = ts.Module()
        main_module_outputs, main_module_inputs = \
            onnx_fence.back_fence().convert(main_graph.outputs, after=main_graph.inputs)
        main_module.load(main_module_outputs)
        main_module.sort_inputs(main_module_inputs)

        if not os.path.isdir(output_root):
            os.makedirs(output_root)

        with open(filename, "wb") as f:
            ts.Module.Save(f, main_module)

        print("[INFO]: Writen file: {}".format(filename))

    def _split_root_name_ext(self, filename):
        # type: (str) -> Tuple[str, str, str]
        filepath = os.path.abspath(filename)
        root, name_ext = os.path.split(filepath)
        name, ext = os.path.splitext(name_ext)
        if len(name) > 1 and name[0] == '.':
            name = name[1:]
        return root, name, ext

    def _pack_node_shapes(self, nodes):
        pack = []
        pack.append(len(nodes))
        for node in nodes:
            shape = node.shape
            pack.append(len(shape))
            pack.extend(shape)
        return pack

    def export_rknn_cfg(self, filename, calibrator=None):
        # type: (str, Calibrator) -> List[RKNNConfig]
        """
        :param filename:
        :param calibrator:
        :return: list of RKNNConfig
        rknn2 operator define:
        rknn2(List[Tensor]) -> List[Tensor]
        attrs:
            `input_count` `Int` `Required`
            `output_count` `Int` `Required`
            `onnx_file` `String` `Required` path to onnx file
            `rknn_file` `String` `Required` path to rknn file
            `rknn_buffer` `ByteArray` `Optional` load from this buffer if rknn_buffer set
            `format` `String` `Optional` default NCHW
            `input_shapes` `List[Int]` `Required`
            `output_shapes` `List[Int]` `Required`
        ``
        """
        if self.__original_module is None:
            raise ValueError("call load fist be before export_rknn_cfg")

        if self.config.do_quantization and calibrator is None:
            raise ValueError("calibrator must be set if do_quantization=True")

        output_root, output_name, output_ext = self._split_root_name_ext(filename)
        # 1. split caffe
        main_graph = split_onnx(self.__original_module, filename, "onnx", self.__input_shape, export_main=False)

        sub_graph_count = main_graph.sub_count()

        dataset_list = None
        if self.config.do_quantization:
            # build dataset list
            print("[INFO]: Building image list... ")
            output_names = []
            for i in range(sub_graph_count):
                sub_graph = main_graph.sub_graph(i)
                output_names.append([node.name for node in sub_graph.inputs])
                print("[INFO]: Dumping {} -> {}".format([node.name for node in sub_graph.inputs], [node.name for node in sub_graph.outputs]))
            dataset_list = export_image_list(module=self.__original_module,
                              output_names=output_names,
                              calibrator=calibrator,
                              main=output_name,
                              output_root=os.path.join(output_root, "data"),
                              cache=self.__cache,
                              device=self.__host_device,
                              device_id=self.__host_device_id)

        summery_configs = []

        # 3. update main graph and set config
        for i in range(sub_graph_count):
            node = main_graph.sub_node(i)
            graph = main_graph.sub_graph(i)

            # ref wk filename
            # rknn_instruction_name = os.path.join("rknn", "{}.{}".format(output_name, i))
            # rknn_filename = rknn_instruction_name + ".rknn"
            # print("[INFO]: Waiting... {}".format(rknn_filename))

            onnx_filename = os.path.join("onnx", "{}.{}.onnx".format(output_name, i))
            rknn_filename = os.path.join("rknn", "{}.{}.rknn".format(output_name, i))

            # update node
            node.op = "rknn2"
            node.set("input_count", len(graph.inputs), numpy.int32)     # required
            node.set("output_count", len(graph.outputs), numpy.int32)   # required
            node.set("onnx_file", onnx_filename)    # optinal
            node.set("rknn_file", rknn_filename)    # required
            node.set("input_shapes", self._pack_node_shapes(graph.inputs), numpy.int32)
            node.set("output_shapes", self._pack_node_shapes(graph.outputs), numpy.int32)
            node.set("format", "NCHW")
            # remain rknn_buffer not set

            cfg = copy.copy(self.__config)
            cfg.onnx_file = os.path.join(output_root, onnx_filename)
            cfg.rknn_file = os.path.join(output_root, rknn_filename)

            cfg.inputs = [node.name for node in graph.inputs]
            cfg.outputs = [node.name for node in graph.outputs]

            # set not change channels means
            assert len(graph.inputs) == 1, "Only support single input for now"
            input_node = graph.inputs[0]
            input_shape = input_node.shape
            assert input_shape is not None and len(input_shape) == 4, "Input set must be set as NCHW format"
            input_channels = input_shape[1]
            assert input_channels > 0, "Input channels must greater than 0"
            if cfg.reorder_channel is None and cfg.channel_mean_value is None:
                cfg.default_channels(input_channels)
            elif cfg.reorder_channel is not None and cfg.channel_mean_value is not None:
                pass
            else:
                raise RuntimeError("config's reorder_channel and channel_mean_value must be both set or both none.")

            if dataset_list is not None:
                cfg.dataset = dataset_list[i]

            summery_configs.append(cfg)

        # 4. write main tsm file
        main_module = ts.Module()
        main_module_outputs, main_module_inputs = \
            onnx_fence.back_fence().convert(main_graph.outputs, after=main_graph.inputs)
        main_module.load(main_module_outputs)
        main_module.sort_inputs(main_module_inputs)

        if not os.path.isdir(output_root):
            os.makedirs(output_root)

        with open(filename, "wb") as f:
            ts.Module.Save(f, main_module)

        print("[INFO]: Writen file: {}".format(filename))

        return summery_configs

    @staticmethod
    def FuseRKNN(input_filename, output_filename):
        # type: (str, str) -> None
        """
        Fuse all nnie operators' wk_file to wk_buffer
        :param input_filename:
        :param output_filename:
        :return:
        """
        input_root = os.path.split(os.path.abspath(input_filename))[0]
        # output_root = os.path.split(os.path.abspath(output_filename))[0]
        with open(input_filename, "rb") as f:
            input_module = ts.Module.Load(f)
        input_graph, _ = ts.graph.walk_graph(input_module.outputs)
        for node in input_graph:
            if node.op == "rknn2":
                wk_file = str(node.get("rknn_file"))
                abs_wk_file = os.path.join(input_root, wk_file)
                if not os.path.isfile(abs_wk_file):
                    raise FileNotFoundError("File {} not found in {}".format(wk_file, input_filename))
                # merge file in wk_file
                with open(abs_wk_file, "rb") as f:
                    wk_buffer = f.read()
                # buffer to tensor
                dtype_numpy = numpy.dtype(numpy.uint8)
                dtype_numpy = dtype_numpy.newbyteorder('<')
                tensor = numpy.frombuffer(wk_buffer, dtype=dtype_numpy)
                tensor = numpy.reshape(tensor, [-1])
                node.set("rknn_buffer", tensor)

        output_module = input_module
        with open(output_filename, "wb") as f:
            ts.Module.Save(f, output_module)

    def suggest_name(self, output_root, input_filename):
        path, name_ext = os.path.split(input_filename)
        name, ext = os.path.splitext(name_ext)
        return os.path.join(output_root, "{}.{}{}".format(name, self.config.tag(), ext))

    def _tmp_filename(self, name):
        path, name = os.path.split(name)
        return os.path.join(path, ".{}".format(name))

    def export_tsm_with_rknn(self, output_filename, calibrator=None, tmp_filename=None, fuse_rknn=True):
        # type: (str, Calibrator, str, bool) -> None

        if tmp_filename is not None:
            hide_filename = tmp_filename
        else:
            hide_filename = self._tmp_filename(output_filename)

        rknn_configs = self.export_rknn_cfg(hide_filename, calibrator)

        for i, cfg in enumerate(rknn_configs):
            try:
                print("[INFO]: Processing [{}/{}] rknn export.".format(i + 1, len(rknn_configs)))
                from . import rknn_api as rknn
                master = rknn.RKNNMaster(original_model=cfg.onnx_file, ignore_buffer=True,
                                         rknn_model=cfg.rknn_file,
                                         target=cfg.target, device_id=cfg.device_id,
                                         do_quantization=cfg.do_quantization,
                                         dataset=cfg.dataset, pre_compile=cfg.pre_compile,
                                         quantized_dtype=cfg.quantized_dtype,
                                         channel_mean_value=cfg.channel_mean_value,
                                         reorder_channel=cfg.reorder_channel,
                                         verbose=cfg.verbose)

                # accuracy_analysis
                if False and cfg.do_quantization and cfg.device_id:
                    analysis_output_dir = os.path.join(os.path.split(cfg.rknn_file)[0], 'snapshot')
                    if not os.path.isdir(analysis_output_dir):
                        os.makedirs(analysis_output_dir)
                    master.accuracy_analysis(cfg.dataset, analysis_output_dir, cfg.device_id)

                # known outputs names
                if False and cfg.inputs and cfg.outputs and calibrator and cfg.device_id:
                    calibrator.reset()
                    extractor = dumper.Dumper(self.__original_module, cfg.inputs + cfg.outputs, calibrator, 1, cache=self.__cache, device=self.__host_device, device_id=self.__host_device_id)
                    features = extractor.next() # test one single input
                    inputs = features[:len(cfg.inputs)]
                    outputs = features[len(cfg.inputs):]
                    # tranpose input from NCHW to NHWC, as it's api feature
                    output_values = master.inference_pass_through(inputs)
                    # print(output_values)
                    # exit(996)
                
                master.release()
            except Exception as _:
                print("[ERROR]: Process {}-th onnx file failed! which is: {}".format(i, cfg.onnx_file))
                raise

        print("[INFO]: Process {} rknn config(s) done.".format(len(rknn_configs)))
        if fuse_rknn:
            self.FuseRKNN(hide_filename, output_filename)
            if hide_filename != tmp_filename:
                os.remove(hide_filename)
        else:
            if hide_filename != tmp_filename:
                os.rename(hide_filename, output_filename)
            else:
                import shutil
                shutil.copy(hide_filename, output_filename)

        print("[INFO]: Writen file: {}".format(output_filename))
