#!/usr/bin/env python

from rknn.api import RKNN

import cv2
import numpy as np
import time
import os


from typing import List, Tuple, Optional, Union


def timestr():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))


def _tell_one_model(model):
    # type: (str) -> str
    assert isinstance(model, str)
    name, ext = os.path.splitext(model)
    if ext == ".onnx":
        return "onnx"
    raise Exception("Can not tell original framework: {}".format(model))


def _tell_witch_model(model):
    # type: (List[str]) -> str
    if len(model) == 1:
        return _tell_one_model(model[0])
    raise Exception("Can not tell original framework: {}".format(model))


class RKNNMaster(object):
    def __init__(self, original_model=None, rknn_model=None, buffer_root=None, ignore_buffer=False,
                 target=None, device_id=None,
                 do_quantization=False,
                 dataset=None,
                 pre_compile=True,
                 quantized_dtype=None,
                 channel_mean_value=None, reorder_channel=None,
                 verbose=False):
        # type: (Union[str, List[str]], str, str, bool, str, str, bool, str, bool, Optional[str], Union[str, List[int]], Union[str, List[int]], bool) -> None

        if pre_compile:
            print('[INFO]', 'ignore parameter pre_compile=True')

        if isinstance(channel_mean_value, (list, tuple)):
            channel_mean_value = ' '.join([str(i) for i in channel_mean_value])
        if isinstance(reorder_channel, (list, tuple)):
            reorder_channel = ' '.join([str(i) for i in reorder_channel])

        mean_values = None
        std_values = None
        if isinstance(channel_mean_value, str):
            values = [float(v) for v in channel_mean_value.split(' ')]
            mean_values = [values[:3]]
            std_values = [values[-1:] * 3]
            print("[{}] Using: mean_values=\"{}\", std_values=\"{}\"".format(timestr(), mean_values, std_values))

        quant_img_RGB2BGR = None
        if isinstance(reorder_channel, str):
            assert reorder_channel in {"0 1 2", "2 1 0"}
            if reorder_channel == "2 1 0":
                quant_img_RGB2BGR = True

        self.__target = target
        self.__device_id = device_id

        kwargs = {"verbose": verbose}

        if buffer_root is None:
            buffer_root = "/tmp/rknn"

        if not os.path.isdir(buffer_root):
            os.makedirs(buffer_root)

        allowed_quantized_dtype = {
            "asymmetric_quantized-u8", 'dynamic_fixed_point-8', 'dynamic_fixed_point-16',
            "asymmetric_quantized-8", "asymmetric_quantized-16", 
        }
        assert quantized_dtype is None or quantized_dtype in allowed_quantized_dtype, \
            "quantized_dtype can be None or in {}".format(allowed_quantized_dtype)

        if original_model is None and rknn_model is None:
            raise Exception('Must set original_model or rknn_model')

        if target is None and device_id is None:
            self.__rknn = RKNN(**kwargs)
        elif device_id is None:
            # self.__rknn = RKNN(target=target)
            self.__rknn = RKNN(**kwargs)
        else:
            # self.__rknn = RKNN(target=target, device_id=device_id)
            self.__rknn = RKNN(**kwargs)
        print("[{}] Init with: target=\"{}\", device_id=\"{}\"".format(timestr(), target, device_id))

        batch_size = 1
        print('[{}] --> config model: channel_mean_value=\"{}\", reorder_channel=\"{}\"'.format(timestr(), channel_mean_value, reorder_channel))
        self.__rknn.config(mean_values=mean_values,
                           std_values=std_values,
                           quant_img_RGB2BGR=quant_img_RGB2BGR,
                           quantized_dtype=quantized_dtype,
                           optimization_level=0,
                           target_platform=None if target is None else target)

        if isinstance(original_model, str):
            original_model = [original_model]
        assert isinstance(original_model, (tuple, list))

        # assert reorder_channel in {"0 1 2", "2 1 0"}

        fixed_channel_mean_value = "[]" if channel_mean_value is None else channel_mean_value.replace(" ", "_")
        fixed_reorder_channel = "[]" if reorder_channel is None else reorder_channel.replace(" ", "_")
        model_mark = os.path.split(original_model[0])[-1]
        model_mark = os.path.splitext(model_mark)[1]
        option_mark = "{}.{}.{}.{}".format(
            # "precompile" if pre_compile else "no-compile",
            "do-quantization" if do_quantization else "no-quantization",
            quantized_dtype if do_quantization else "float16",
            fixed_channel_mean_value,
            fixed_reorder_channel
        )
        buffer_rknn_model = '{}.{}.rknn'.format(model_mark, option_mark)
        buffer_rknn_model = os.path.join(buffer_root, buffer_rknn_model)

        load_rknn = False
        if rknn_model is not None and not ignore_buffer:
            print('[{}] --> Loading rknn model'.format(timestr(), ))
            ret = self.__rknn.load_rknn(path=rknn_model)
            if ret == 0:
                load_rknn = True
                print('[{}] --> Loading rknn model done.'.format(timestr(), ))
        elif not ignore_buffer:
            # check buffer
            print('[{}] --> Loading rknn model'.format(timestr(), ))
            ret = self.__rknn.load_rknn(path=buffer_rknn_model)
            if ret == 0:
                load_rknn = True
                print('[{}] --> Loading rknn model done.'.format(timestr(), ))

        if device_id is None:
            target = None
            # no device id can not init target runtime

        if load_rknn:
            ret = self.__rknn.init_runtime(target=target, device_id=device_id)
            if ret != 0:
                print('[{}] Init runtime {}:{} failed! ret={}.'.format(timestr(), target, device_id, ret))
                exit(ret)
            return

        if original_model is None:
            raise Exception("[{}] Must set original_model.".format(timestr(), ))

        # Load original model
        model_type = _tell_witch_model(original_model)
        if model_type == "onnx":
            ret = self.__rknn.load_onnx(model=original_model[0])
        else:
            raise Exception("[{}] Not support model type: {}.".format(timestr(), model_type))

        if ret != 0:
            raise Exception("[{}] Can not load {}: {}".format(timestr(), model_type, original_model))
        print('[{}] --> Loading {} model done.'.format(timestr(), model_type))

        # Build model
        print('[{}] --> Building model'.format(timestr(), ))
        ret = self.__rknn.build(do_quantization=do_quantization, dataset=dataset)
        if ret != 0:
            print('[{}] Build {} failed!'.format(timestr(), model_type))
            exit(ret)
        print('[{}] --> Building model done.'.format(timestr(), ))

        ret = self.__rknn.init_runtime(target=target, device_id=device_id)
        if ret != 0:
            print('[{}] Init runtime {}:{} failed! ret={}.'.format(timestr(), target, device_id, ret))
            exit(ret)

        if rknn_model is not None:
            rknn_model_root = os.path.split(rknn_model)[0]
            if not os.path.isdir(rknn_model_root):
                os.makedirs(rknn_model_root)
            print('[{}] --> Exporting model'.format(timestr(), ))
            self.__rknn.export_rknn(export_path=rknn_model)
            print('[{}] --> Exporting model done.'.format(timestr(), ))

        self.__rknn.export_rknn(export_path=buffer_rknn_model)

    def export(self, rknn_model):
        rknn_model_root = os.path.split(rknn_model)[0]
        if not os.path.isdir(rknn_model_root):
            os.makedirs(rknn_model_root)
        print('[{}] --> Exporting model'.format(timestr(), ))
        self.__rknn.export_rknn(export_path=rknn_model)
        print('[{}] --> Exporting model done.'.format(timestr(), ))

    def inference(self, opencv_image):
        """
        :param opencv_image opencv's image, in HWC format and BGR layout
        """
        img = opencv_image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.asarray(img, dtype=float) / 255.0
        # img = np.transpose(img, (2, 0, 1))
        outputs = self.__rknn.inference(inputs=[img])
        output = outputs

        return output

    def accuracy_analysis(self, images, output_dir):
        """
        :param images str or list of str
        :param output_dir output dir
        """

        tmp_root = "/tmp/rknn/"
        tmp_file = tmp_root + "tmp_accuracy_analysis_dataset.txt"
        if not os.path.isdir(tmp_root):
            os.makedirs(tmp_root)
        if isinstance(images, str):
            images = [images]
        assert isinstance(images, (tuple, list))
        with open(tmp_file, "w") as f:
            for image in images:
                f.write(image)
                f.write(" ")
            f.write("\n")

        ret = self.__rknn.accuracy_analysis(inputs=tmp_file, output_dir=output_dir, calc_qnt_error=True)
        if ret != 0:
            print('[{}] Accuracy analysis failed. ret={}.'.format(timestr(), ret))

    def log(self):
        print("Run duration: {}us".format(self.__rknn.get_run_duration()))

    def release(self):
        self.__rknn.release()


def compare(e1, e2, epsilon=1e-5):
    _y1 = np.sqrt(np.sum(np.square(e1), axis=-1))
    _y2 = np.sqrt(np.sum(np.square(e2), axis=-1))
    y1_y2 = np.sum(np.multiply(e1, e2), axis=-1)
    cosine = y1_y2 / (_y1 * _y2 + epsilon)
    return cosine


def diff(e1, e2):
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


if __name__ == '__main__':
    """
    Older onnx verion: 1.4.1
    """
    CWD = "/home/kier/Documents/rknn-toolkit-v0.9.3/rknn_example/onnx/"

    image = CWD + './arcface_112x112/data/fr_crop_112x112_2k/00000.jpg'

    target = 'rk3399pro'
    device_id = 'TUXG0IBTRB'

    onnx_model = CWD + './arcface_112x112/model_arcface_2020-3-4.onnx'
    output_root = CWD + './arcface_112x112/rknn'
    dataset = CWD + "./arcface_112x112/data/dataset.txt"

    pre_compile = True

    do_quantization = False
    quantized_dtype = 'asymmetric_quantized-u8'
    # quantized_dtype = 'dynamic_fixed_point-8'
    # quantized_dtype = 'dynamic_fixed_point-16'

    model_mark = os.path.splitext(os.path.split(onnx_model)[-1])[0]

    option_mark = "{}.{}.{}".format(
        "precompile" if pre_compile else "no-compile",
        "do-quantization" if do_quantization else "no-quantization",
        quantized_dtype if do_quantization else "float16"
    )
    rknn_model = '{}/{}.{}.rknn'.format(output_root, model_mark, option_mark)

    rknn = RKNNMaster(target=target, device_id=device_id,
                      original_model=onnx_model, rknn_model=rknn_model,
                      pre_compile=pre_compile,
                      quantized_dtype=quantized_dtype,
                      do_quantization=do_quantization, dataset=dataset,
                      channel_mean_value=[0, 0, 0, 255], reorder_channel=[2, 1, 0])

    # test
    img = cv2.imread(image)

    target_features_file = '79w_features/fc1_act_output.txt'

    print('--> Running model')
    outputs = rknn.inference(img)
    print('--> Running model done.')

    features = outputs[0][0]

    print(features.shape)
    print(features)

    rknn.release()
