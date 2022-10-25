#!/usr/bin/env python

from rknn.api import RKNN
import cv2
import numpy as np
import time
import os


def timestr():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))


def analysis_dataset(dataset):
    output = dataset + "._first_line_.txt"
    line = ""
    with open(dataset, "r") as f:
        line = f.readline()
    with open(output, "w") as f:
        f.write(line)
    return output


def do_quant():
    pre_compile = False
    do_quantization = True
    quantized_dtype = 'asymmetric_quantized-u8'
    # quantized_dtype = 'dynamic_fixed_point-8'
    # quantized_dtype = 'dynamic_fixed_point-16'

    kwargs = {"verbose": True}
    batch_size = 100

    target = "rk3399pro"
    device_id = None

    root = "/home/kier/working/pose/rknn/"
    tag = "pose.rk3399pro.[].[].no-compile.do-quantization.asymmetric_quantized-u8.0"

    dataset = root + "data/pose.out_sigmoid.in360.rk3399pro.[].[].no-compile.do-quantization.asymmetric_quantized-u8/0_[input.1].txt"
    onnx = root + "onnx/{}.onnx".format(tag)
    output = root +"rknn/{}.rknn".format(tag)
    analysis_dir = "snapshot"

    rknn = RKNN(**kwargs)
    print("[{}] Init with: target=\"{}\", device_id=\"{}\"".format(timestr(), target, device_id))

    rknn.config(channel_mean_value='0 0 0 1', reorder_channel='0 1 2', batch_size=batch_size, epochs=-1)

    ret = rknn.load_onnx(model=onnx)
    if ret != 0:
        rknn.release()
        raise Exception("[{}] Can not load ONNX: {}".format(timestr(), onnx))

    ret = rknn.build(do_quantization=do_quantization, dataset=dataset, pre_compile=pre_compile)
    if ret != 0:
        rknn.release()
        raise Exception('[{}] Build onnx failed!'.format(timestr(), ))

    ret = rknn.export_rknn(export_path=output)
    if ret != 0:
        rknn.release()
        raise Exception("[{}] Can not export rknn with ret = {}".format(timestr(), ret))

    ret = rknn.init_runtime(target=target, device_id=device_id)
    if ret != 0:
        rknn.release()
        raise Exception("[{}] Can not init runtime with ret = {}".format(timestr(), ret))

    print("[{}] Start analysis...".format(timestr()))

    if not os.path.isdir(analysis_dir):
        os.makedirs(analysis_dir)
    rknn.accuracy_analysis(inputs=analysis_dataset(dataset), output_dir=analysis_dir)

    print("[{}] Export rknn: \"{}\"".format(timestr(),output))

    rknn.release()


def do_fuse():
    root = "/home/kier/working/pose/rknn/"
    tag = "pose.rk3399pro.[].[].no-compile.do-quantization.asymmetric_quantized-u8"

    tsm = root + "{}.tsm".format(tag)
    output = root + "{}.hybrid.tsm".format(tag)

    import tennis_rknn as rknn
    rknn.exporter.RKNNExporter.FuseRKNN(tsm, output)

    print("[{}] Fuse tsm: \"{}\"".format(timestr(),output))


def do_hybrid_step1():
    pre_compile = False
    do_quantization = True
    quantized_dtype = 'asymmetric_quantized-u8'
    # quantized_dtype = 'dynamic_fixed_point-8'
    # quantized_dtype = 'dynamic_fixed_point-16'

    kwargs = {"verbose": True}
    batch_size = 1

    target = "rk3399pro"
    device_id = None

    root = "/home/kier/working/pose/rknn/"
    tag = "pose.rk3399pro.[].[].no-compile.do-quantization.asymmetric_quantized-u8.0"

    dataset = root + "data/pose.origin_and_oh360.txt"
    # dataset = root + "data/pose.oh360.rk3399pro.[].[].no-compile.do-quantization.asymmetric_quantized-u8/0_[input.1].txt"
    # dataset = root + "data/pose.out_sigmoid.in360.rk3399pro.[].[].no-compile.do-quantization.asymmetric_quantized-u8/0_[input.1].txt"
    # dataset = root + "data/pose.rk3399pro.[].[].no-compile.do-quantization.asymmetric_quantized-u8/0_[input.1].txt"
    onnx = root + "onnx/{}.onnx".format(tag)
    output = root +"rknn/{}.rknn".format(tag)

    rknn = RKNN(**kwargs)
    print("[{}] Init with: target=\"{}\", device_id=\"{}\"".format(timestr(), target, device_id))

    rknn.config(channel_mean_value='0 0 0 1', reorder_channel='0 1 2', batch_size=batch_size, epochs=-1)

    ret = rknn.load_onnx(model=onnx)
    if ret != 0:
        rknn.release()
        raise Exception("[{}] Can not load ONNX: {}".format(timestr(), onnx))

    ret = rknn.hybrid_quantization_step1(dataset=dataset)
    if ret != 0:
        rknn.release()
        raise Exception("[{}] Call hybrid_quantization_step1 faile with ret={}".format(timestr(), ret))

    print("[{}] Export data json and cfg in current folder: \"{}\"".format(timestr(), os.getcwd()))

    rknn.release()


def do_hybrid_step2():
    pre_compile = False
    do_quantization = True
    quantized_dtype = 'asymmetric_quantized-u8'
    # quantized_dtype = 'dynamic_fixed_point-8'
    # quantized_dtype = 'dynamic_fixed_point-16'

    kwargs = {"verbose": True}
    batch_size = 1

    target = "rk3399pro"
    device_id = "07SS4Z8H30"

    root = "/home/kier/working/pose/rknn/"
    tag = "pose.rk3399pro.[].[].no-compile.do-quantization.asymmetric_quantized-u8.0"
    cfg_prefix = "poserk3399pronocompiledoquantizationasymmetric_quantizedu80onnx"
    analysis_dir = "snapshot"

    example = root + "data/pose.debug.rk3399pro.[].[].no-compile.do-quantization.asymmetric_quantized-u8/0_[input.1].txt"
    dataset = root + "data/pose.origin_and_oh360.txt"
    # dataset = root + "data/pose.oh360.rk3399pro.[].[].no-compile.do-quantization.asymmetric_quantized-u8/0_[input.1].txt"
    # dataset = root + "data/pose.out_sigmoid.in360.rk3399pro.[].[].no-compile.do-quantization.asymmetric_quantized-u8/0_[input.1].txt"
    # dataset = root + "data/pose.rk3399pro.[].[].no-compile.do-quantization.asymmetric_quantized-u8/0_[input.1].txt"
    # onnx = root + "onnx/{}.onnx".format(tag)
    output = root +"rknn/{}.rknn".format(tag)

    model_input = "{}.json".format(cfg_prefix)
    data_input = "{}.data".format(cfg_prefix)
    model_quantization_cfg = "{}.quantization.cfg".format(cfg_prefix)

    rknn = RKNN(**kwargs)
    print("[{}] Init with: target=\"{}\", device_id=\"{}\"".format(timestr(), target, device_id))

    rknn.config(channel_mean_value='0 0 0 1', reorder_channel='0 1 2', batch_size=batch_size, epochs=-1)

    ret = rknn.hybrid_quantization_step2(model_input=model_input,
                                         data_input=data_input,
                                         model_quantization_cfg=model_quantization_cfg,
                                         dataset=dataset,
                                         pre_compile=pre_compile)
    if ret != 0:
        rknn.release()
        raise Exception("[{}] Call hybrid_quantization_step2 faile with ret={}".format(timestr(), ret))

    ret = rknn.export_rknn(export_path=output)
    if ret != 0:
        rknn.release()
        raise Exception("[{}] Can not export rknn with ret = {}".format(timestr(), ret))

    # do_fuse()
    #
    # ret = rknn.init_runtime(target=target, device_id=device_id)
    # if ret != 0:
    #     rknn.release()
    #     raise Exception("[{}] Can not init runtime with ret = {}".format(timestr(), ret))
    #
    # print("[{}] Start analysis...".format(timestr()))
    #
    # if not os.path.isdir(analysis_dir):
    #     os.makedirs(analysis_dir)
    # rknn.accuracy_analysis(inputs=analysis_dataset(example), output_dir=analysis_dir)

    print("[{}] Export rknn: \"{}\"".format(timestr(),output))

    rknn.release()



if __name__ == '__main__':
    # do_quant()
    # do_fuse()

    # do_hybrid_step1()
    do_hybrid_step2()
    do_fuse()