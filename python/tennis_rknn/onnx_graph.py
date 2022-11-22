# coding: UTF-8

import tennis as ts

from typing import List
import os
from collections import OrderedDict
import numpy

from tennisfence.fence import Fence
from tennisfence.metanode import *

from typing import Optional
import onnx
import tennisbuilder as tb
import onnx.shape_inference


def fuse_conv2d_bias(node):
    # type: (ts.Node) -> ts.Node
    conv2d = ts.graph.clone_bubble(node.inputs[0])
    conv2d.name = node.name
    ts.Node.Link(conv2d, (node.inputs[0].inputs[0], node.inputs[0].inputs[1], node.inputs[1]))
    return conv2d


def fuse_ip_bias(node):
    # type: (ts.Node) -> ts.Node
    ip = ts.graph.clone_bubble(node.inputs[0])
    ip.name = node.name
    ts.Node.Link(ip, (node.inputs[0].inputs[0], node.inputs[0].inputs[1], node.inputs[1]))
    return ip


def transpose_weights(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]
    w = node.inputs[1]
    b = node.inputs[2] if len(node.inputs) > 2 else None

    transpose = bool(node.try_get("transpose", False))
    if transpose:
        return None

    w_t = numpy.transpose(ts.zoo.to_const(w))
    w_t = ts.menu.data(name=w.name + "_t", value=w_t)

    ip = ts.graph.clone_bubble(node)
    ip.name = node.name
    ip.set("transpose", True)

    inputs = [x, w_t, b] if b is not None else [x, w_t]

    ts.Node.Link(ip, inputs)

    return ip


def check_if_3_inputs(node):
    # type: (ts.Node) -> Optional[ts.Node]
    if len(node.inputs) > 2:
        return None

    x = node.inputs[0]
    w = node.inputs[1]

    transpose = bool(node.try_get("transpose", False))
    if transpose:
        w_t = numpy.transpose(ts.zoo.to_const(w))
        w_t = ts.menu.data(name=w.name + "_t", value=w_t)
        w = w_t

    mat_mul = ts.menu.op(name=node.name, op_name="mat_mul", inputs=[x, w])

    if node.has("#dtype"):
        dtype = node.dtype
        mat_mul.dtype = dtype

    mat_mul.shape = node.shape

    return mat_mul


def ensure_has_bias(node):
    # type: (ts.Node) -> Optional[ts.Node]
    if len(node.inputs) > 2:
        return None

    b = ts.menu.data(node.name + "_b", numpy.asarray(0, dtype=numpy.float32), device=ts.device.CPU)

    node3 = ts.graph.clone_bubble(node)
    ts.Node.Link(node3, [node.inputs[0], node.inputs[1], b])

    return node3


def fuse_flatten_ip(node):
    # type: (ts.Node) -> Optional[ts.Node]
    flatten = node.inputs[0]
    ip = node
    w = node.inputs[1]

    if flatten.has("dim") and int(flatten.get("dim")) != 1:
        return None

    new_ip = ts.graph.clone_bubble(ip)
    ts.Node.Link(new_ip, (flatten.inputs[0], w))

    return new_ip


def fuse_bias_reshape(node):
    # type: (ts.Node) -> Optional[ts.Node]
    bias = node.inputs[0]
    reshape = node

    shape = list(reshape.get("shape"))
    if len(shape) != 4 or shape[2] != 1 or shape[3] != 1:
        return None

    return bias


def change_sub_neg(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[1]

    return ts.menu.op(node.name, "neg", [x])


def change_relu6_x4(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]
    name = node.name

    channels = node.shape[1]
    max = float(node.get("max"))

    # change relu6 to x - (x - 6) * thresh(x, 6)
    relu = ts.zoo.relu(name + "_relu_", x)
    bias = ts.zoo.add_bias(name + "_bias_", relu, b=[-max, ] * channels, dim=1)
    thresh = ts.menu.op(name + "_thresh_", "threshold", [relu])
    thresh.set("threshold", max, numpy.float32)
    bias_x_thresh = ts.zoo.mul(name + "_bias_x_thresh", bias, thresh)
    relu6_x4 = ts.zoo.sub(name, relu, bias_x_thresh)

    if node.has("#dtype"):
        dtype = node.dtype
        relu.dtype = dtype
        bias.dtype = dtype
        thresh.dtype = dtype
        bias_x_thresh.dtype = dtype
        relu6_x4.dtype = dtype

    relu.shape = node.shape
    bias.shape = node.shape
    thresh.shape = node.shape
    bias_x_thresh.shape = node.shape
    relu6_x4.shape = node.shape

    return relu6_x4


def change_reshape_v2(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]
    name = node.name

    shape = node.get("shape")
    shape_node = ts.menu.data(name=name + "_shape", value=shape)

    reshape_v2 = ts.zoo.reshape_v2(name=name, x=x, shape=shape_node, force=True)

    if node.has("#dtype"):
        dtype = node.dtype
        reshape_v2.dtype = dtype

    reshape_v2.shape = node.shape

    return reshape_v2


def flatten2reshape(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]
    name = node.name

    dim = int(node.get("dim"))

    shape = numpy.asarray([0] * dim + [-1], dtype=numpy.int32)
    shape_node = ts.menu.data(name=name + "_shape", value=shape)

    reshape_v2 = ts.zoo.reshape_v2(name=name, x=x, shape=shape_node, force=True)

    if node.has("#dtype"):
        dtype = node.dtype
        reshape_v2.dtype = dtype

    reshape_v2.shape = node.shape

    return reshape_v2


def change_reshape_v2_int64(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]
    shape = node.inputs[1]

    name = node.name

    shape_int64_value = numpy.asarray(shape.get("value"), dtype=numpy.int64)

    shape_node = ts.menu.data(name=name + "_shape", value=shape_int64_value)

    onnx_reshape = ts.menu.op(name, "onnx::reshape", [x, shape_node])

    if node.has("#dtype"):
        dtype = node.dtype
        onnx_reshape.dtype = dtype

    onnx_reshape.shape = node.shape

    return onnx_reshape


def change_tile_v2_int64(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]
    repeats = node.inputs[1]

    name = node.name

    repeats_int64_value = numpy.asarray(repeats.get("value"), dtype=numpy.int64)

    repeats_node = ts.menu.data(name=name + "_repeats", value=repeats_int64_value)

    onnx_tile = ts.menu.op(name, "onnx::tile", [x, repeats_node])

    if node.has("#dtype"):
        dtype = node.dtype
        onnx_tile.dtype = dtype

    onnx_tile.shape = node.shape

    return onnx_tile


def fuse_batch_nrom(node):
    # type: (ts.Node) -> Optional[ts.Node]
    batch_scale = node
    batch_norm = node.inputs[0]

    x = batch_norm.inputs[0]
    mean = batch_norm.inputs[1]
    var = batch_norm.inputs[2]
    scale = batch_scale.inputs[1]
    bias = batch_scale.inputs[2]

    dim1 = int(batch_norm.try_get("dim", 1))
    dim2 = int(batch_scale.try_get("dim", 1))
    epsilon = batch_norm.try_get("1e-05", 1e-5)

    if dim1 != dim2:
        return None

    if dim1 not in {1}:
        return None

    name = node.name

    fused = ts.zoo.fused_batch_norm(name=name, x=x,
                                    mean=mean,
                                    variance=var,
                                    scale=scale,
                                    bias=bias,
                                    dim=dim1,
                                    epsilon=epsilon)

    if node.has("#dtype"):
        dtype = node.dtype
        fused.dtype = dtype

    fused.shape = node.shape

    return fused


def change_dim_shuffle_gather(node):
    # type: (ts.Node) -> Optional[ts.Node]

    x = node.inputs[0]
    dim = node.get("dim")
    shuffle = node.get("shuffle")

    name = node.name

    shuffle_node = ts.menu.data(name + "_indices", shuffle)

    gather = ts.graph.clone_bubble(node)
    gather.op = "gather"
    gather.clear("dim")
    gather.set("axis", dim, numpy.int64)

    ts.Node.Link(gather, [x, shuffle_node])

    return gather


def change_gather_indices_int64(node):
    # type: (ts.Node) -> Optional[ts.Node]

    x = node.inputs[0]
    indices = node.inputs[1]

    indices_data = ts.zoo.to_const(indices)
    if indices_data.dtype == numpy.int64:
        return None

    indices_data = numpy.asarray(indices_data, numpy.int64)
    indices = ts.menu.data(indices.name + "_int64", indices_data)

    gather_int64 = ts.graph.clone_bubble(node)
    ts.Node.Link(gather_int64, [x, indices])

    return gather_int64


def change_softmax_suitable(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]

    dim = int(node.get("dim"))
    shape = list(node.shape)
    size = len(shape)

    if dim == -1 or dim == size - 1:
        return None

    if numpy.prod(shape[dim + 1:]) == 1:
        return None

    if dim < 0:
        dim += size

    transpose1 = list(range(size))
    del transpose1[dim]
    transpose1.append(dim)

    transpose_shape1 = numpy.asarray(shape)[transpose1]

    transpose2 = list(range(size))
    del transpose2[-1]
    transpose2.insert(dim, size - 1)

    transpose_shape2 = numpy.asarray(transpose_shape1)[transpose2]

    core = ts.graph.clone_bubble(node)

    core.name = node.name + "_core"
    core.set("dim", -1)

    node1 = ts.zoo.transpose(name=node.name + "_t1", x=x, permute=transpose1)
    ts.Node.Link(core, [node1])
    node2 = ts.zoo.transpose(name=node.name, x=core, permute=transpose2)

    if node.has("#dtype"):
        dtype = node.dtype
        node1.dtype = dtype
        core.dtype = dtype
        node2.dtype = dtype

    node1.shape = transpose_shape1
    core.shape = transpose_shape1
    node2.shape = shape

    return node2


def change_broadcast_add(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]

    assert node.inputs[1].op == ts.Node.Const

    y_shape = ts.inferer.infer_value(node.inputs[1])
    y_dtype = ts.dtype.FLOAT32
    if x.has("#dtype"):
        y_dtype = int(x.dtype)

    y_node = ts.menu.data(name=node.name + "_zero",
                          value=numpy.zeros(y_shape, dtype=ts.dtype.to_numpy(y_dtype)))

    add_node = ts.zoo.add(name=node.name, lhs=x, rhs=y_node)

    if node.has("#dtype"):
        dtype = node.dtype
        add_node.dtype = dtype

    add_node.shape = node.shape

    return add_node


def convert_depthwise2group(node):
    # type: (ts.Node) -> Optional[ts.Node]
    W = node.inputs[1].get("value")

    number_filters = W.shape[0]
    input_channels = W.shape[1]
    output_channels = number_filters * input_channels

    W = numpy.transpose(W, (1, 0, 2, 3))    # change number filter to dim 1
    W = numpy.reshape(W, [output_channels, 1, W.shape[2], W.shape[3]])

    W_group = ts.menu.data(name=node.inputs[1].name + "_group", value=W)

    inputs = list(node.inputs)
    inputs[1] = W_group

    group_conv2d = ts.graph.clone_bubble(node)
    group_conv2d.op = "group_conv2d"
    group_conv2d.set("group", input_channels, dtype=numpy.int32)

    ts.Node.Link(group_conv2d, inputs)

    return group_conv2d


def change_tile_v2(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]
    name = node.name

    repeats = node.get("repeats")
    repeats_node = ts.menu.data(name=name + "_repeats", value=repeats)

    tile_v2 = ts.menu.op(name=name, op_name="tile_v2", inputs=[x, repeats_node])

    if node.has("#dtype"):
        dtype = node.dtype
        tile_v2.dtype = dtype

    tile_v2.shape = node.shape

    return tile_v2


def change_tile_v2_int64(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]
    repeats = node.inputs[1]

    repeats_value = numpy.asarray(repeats.get("value"))
    if repeats_value.dtype == numpy.int32:
        return None

    name = node.name

    repeats_int64_value = numpy.asarray(repeats.get("value"), dtype=numpy.int64)

    repeats_node = ts.menu.data(name=name + "_repeats", value=repeats_int64_value)

    onnx_tile = ts.menu.op(name=name, op_name="tile_v2", inputs=[x, repeats_node])

    if node.has("#dtype"):
        dtype = node.dtype
        onnx_tile.dtype = dtype

    onnx_tile.shape = node.shape

    return onnx_tile


def pad_node(name, x, padding):
    # type: (ts.Node) -> Optional[ts.Node]
    node = ts.zoo.pad(name=name, x=x, padding=padding)

    if x.has("#dtype"):
        dtype = x.dtype
        node.dtype = dtype

    if x.has("#shape"):
        shape = list(x.shape)
        padding = numpy.asarray(padding).reshape([-1, 2])
        assert padding.shape[0] == len(shape)
        for i in range(len(shape)):
            shape[i] += padding[i, 0] + padding[i, 1]
        node.shape = shape

    return node


def change_to_space2depth(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]
    name = node.name

    padding = numpy.asarray(node.get("padding")).reshape([2, 2])
    block_shape = numpy.asarray(node.get("block_shape")).reshape([2])

    batch = block_shape[0] * block_shape[1]

    padding4d = numpy.zeros([4, 2], dtype=numpy.int32)
    padding4d[2:] = padding
    pad_x = pad_node(name=name + "_pad", x=x, padding=padding4d)

    space2depth = ts.graph.clone_bubble(node)
    space2depth.op = "space_to_depth4d"
    space2depth.name = name + "_depth"
    space2depth.clear("padding")
    ts.Node.Link(space2depth, pad_x)
    if node.has("#shape"):
        node_shape = list(node.shape)
        space2depth.shape = [1, node_shape[0] * node_shape[1], node_shape[2], node_shape[3]]

    to_batch = ts.zoo.reshape(name, space2depth, [batch, -1, 0, 0])
    if node.has("#dtype"):
        dtype = node.dtype
        to_batch.dtype = dtype

    to_batch.shape = node.shape

    return to_batch


def change_to_depth2space(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]
    name = node.name

    # TODO: default onnx::DepthToSpace DCR mode

    crop = numpy.asarray(node.get("crop")).reshape([2, 2])
    block_shape = numpy.asarray(node.get("block_shape")).reshape([2])

    # assert numpy.all(crop == 0)

    # batch = block_shape[0] * block_shape[1]

    to_depth = ts.zoo.reshape(name + "_depth", x, [1, -1, 0, 0])
    if node.has("#dtype"):
        dtype = node.dtype
        to_depth.dtype = dtype

    if node.has("#shape"):
        x_shape = list(x.shape)
        to_depth.shape = [1, x_shape[0] * x_shape[1], x_shape[2], x_shape[3]]

    depth2space = ts.graph.clone_bubble(node)
    depth2space.op = "depth_to_space4d"
    ts.Node.Link(depth2space, to_depth)

    if not numpy.all(crop == 0):
        if node.has("#shape"):
            before_crop_shape = list(node.shape)
            if before_crop_shape[2] > 0:
                before_crop_shape[2] += crop[0, 0] + crop[0, 1]
            if before_crop_shape[3] > 0:
                before_crop_shape[3] += crop[1, 0] + crop[1, 1]
            depth2space.shape = before_crop_shape
        crop_padding = numpy.zeros([4, 2], dtype=numpy.int32)
        crop_padding[2:, :] = -crop
        crop_depth2space = pad_node(name, depth2space, crop_padding)
        depth2space.name = name + "_core"
        depth2space = crop_depth2space

    return depth2space


def change_onnx_pad_v11(node):
    # type: (ts.Node) -> Optional[ts.Node]

    x = node.inputs[0]
    padding = node.inputs[1]

    padding_data = ts.inferer.infer_value(padding)
    padding_data = numpy.transpose(numpy.asarray(padding_data).reshape([-1, 2])).reshape([-1])
    padding_data = numpy.asarray(padding_data, numpy.int64)

    padding = ts.menu.data(padding.name + "_int64", padding_data)

    pad_int64 = ts.graph.clone_bubble(node)
    pad_int64.op = "onnx::pad-11"
    ts.Node.Link(pad_int64, [x, padding])

    return pad_int64


def change_onnx_pad_v2(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]
    padding = node.inputs[1]

    name = node.name

    padding_data = ts.inferer.infer_value(padding)
    padding_data = numpy.transpose(numpy.asarray(padding_data).reshape([-1, 2])).reshape([-1])
    padding_data = numpy.asarray(padding_data, numpy.int32)

    pad_v1 = ts.menu.op(name=name, op_name="onnx::pad-2", inputs=[x])
    pad_v1.set("padding", padding_data, numpy.int32)

    if node.has("#dtype"):
        dtype = node.dtype
        pad_v1.dtype = dtype

    pad_v1.shape = node.shape

    return pad_v1


def change_onnx_resize2unsample(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]
    size = ts.inferer.infer_value(node.inputs[1])
    assert size is not None

    name = node.name

    x_shape = x.shape
    y_shape = node.shape
    if x_shape is None or y_shape is None:
        return None

    x_shape = numpy.asarray(x_shape, dtype=numpy.float32)
    y_shape = numpy.asarray(y_shape, dtype=numpy.float32)
    y_shape += 0.001  # for ceil may failed bug

    scales = y_shape / x_shape
    scales[size <= 0] = 1

    scales = numpy.asarray(scales, dtype=numpy.float32)
    # assert numpy.all(scales >= 1.0)
    # RKNN may support downsample

    scales_node = ts.menu.data(name=name + "_scales", value=scales, device=ts.device.CPU)

    unsample = ts.graph.clone_bubble(node)
    unsample.op = "onnx::unsample"
    ts.Node.Link(unsample, [x, scales_node])

    return unsample


def fix_flatten2d(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]

    dim = int(node.try_get("dim", 1))

    if dim == 1:
        return None

    shape = node.shape
    assert shape is not None

    neg_count = 0
    for dim in shape:
        if dim < 0:
            neg_count += 1
    if neg_count > 2:
        raise Exception("Can not convert this flatten to onnx version")

    reshape = ts.graph.clone_bubble(node)
    reshape.op = "_reshape"
    reshape.clear("dim")
    reshape.set("shape", shape, numpy.int32)
    ts.Node.Link(reshape, [x])

    return reshape


def change_add_bias_to_add(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]
    bias = node.inputs[1]

    if x.shape is None:
        return None

    if bias.op != ts.Node.Const and bias.shape is None:
        return None

    dim = int(node.get("dim"))
    shape = [1, ] * len(x.shape)
    shape[dim] = -1

    reshape_bias = ts.zoo.reshape(name=bias.name + "_reshape", x=bias, shape=shape)

    add = ts.zoo.add(name=node.name, lhs=x, rhs=reshape_bias)

    if node.has("#dtype"):
        dtype = node.dtype
        reshape_bias.dtype = dtype
        add.dtype = dtype

    reshape_bias_shape = list(shape)
    reshape_bias_shape[dim] = x.shape[dim]

    reshape_bias.shape = reshape_bias_shape
    add.shape = node.shape

    return add


def change_prelu_slope(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]
    slope = node.inputs[1]

    dim = int(node.get("dim"))
    assert dim == 1

    slope_data = ts.inferer.infer_value(slope)
    slope_data = numpy.reshape(slope_data, [-1, 1, 1])

    slope_reshape = ts.menu.data(name=slope.name + "_reshape", value=slope_data)

    prelu_onnx = ts.graph.clone_bubble(node)
    prelu_onnx.op = "onnx::prelu"

    ts.Node.Link(prelu_onnx, [x, slope_reshape])

    return prelu_onnx


def change_sample2onnx(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]

    name = node.name
    dim = int(node.get("dim"))
    scale = float(node.get("scale"))

    x_shape = x.shape
    if dim < 0:
        dim += len(x_shape)

    scales = [1] * len(x_shape)
    scales[dim:dim+2] = [scale, scale]

    scales = numpy.asarray(scales, dtype=numpy.float32)
    # assert numpy.all(scales >= 1.0)
    # RKNN may support downsample

    scales_node = ts.menu.data(name=name + "_scales", value=scales, device=ts.device.CPU)

    unsample = ts.graph.clone_bubble(node)
    unsample.op = "onnx::unsample"
    ts.Node.Link(unsample, [x, scales_node])

    return unsample


def _get_onnx_fence(version=9):
    fence = Fence()
    fence.register(MetaGraph([
        "conv2d",
        ("add_bias", -1)
    ]), fuse_conv2d_bias)
    fence.register(MetaGraph([
        "depthwise_conv2d",
        ("add_bias", -1)
    ]), fuse_conv2d_bias)
    fence.register(MetaGraph([
        "inner_prod",
        ("add_bias", -1)
    ]), fuse_ip_bias)
    fence.register(MetaGraph([
        "depthwise_conv2d"
    ]), convert_depthwise2group)
    fence.register(MetaGraph([
        "group_conv2d",
        ("add_bias", -1)
    ]), fuse_conv2d_bias)
    # fence.register(MetaGraph([
    #     "inner_prod",
    # ]), check_if_3_inputs)
    fence.register(MetaGraph([
        "inner_prod",
    ]), transpose_weights)
    fence.register(MetaGraph([
        "inner_prod",
    ]), ensure_has_bias)    # onnx only support has bias
    # fence.register(MetaGraph([
    #     "flatten",
    #     ("inner_prod", -1)
    # ]), fuse_flatten_ip)
    # fence.register(MetaGraph([
    #     "add_bias",
    #     ("_reshape", -1)
    # ]), fuse_bias_reshape)
    # fence.register(MetaGraph([
    #     {"#op": ts.Node.Const, "value": EQ(0)},
    #     ({"#op": "sub", "#shape": HasShape(4)}, {0: -1})
    # ]), change_sub_neg)
    # fence.register(MetaNode(
    #     "relu_max"
    # ), change_relu6_x4)
    fence.register(MetaNode(
        "_reshape"
    ), change_reshape_v2)
    # fence.register(MetaNode(
    #     "flatten"
    # ), flatten2reshape)
    fence.register(MetaNode(
        "_reshape_v2"
    ), change_reshape_v2_int64)
    fence.register(MetaGraph([
        ts.Node.Const,
        ts.Node.Const,
        ts.Node.Const,
        ts.Node.Const,
        ("batch_norm", {1: ABS(0), 2: ABS(1)}),
        ("batch_scale", {0: -1, 1: ABS(2), 2: ABS(3)})
    ]), fuse_batch_nrom)
    fence.register(MetaNode(
        "_dimshuffle"
    ), change_dim_shuffle_gather)
    fence.register(MetaNode(
        "gather"
    ), change_gather_indices_int64)
    fence.register(MetaNode(
        "softmax"
    ), change_softmax_suitable)
    fence.register(MetaNode(
        "broadcast"
    ), change_broadcast_add)
    fence.register(MetaNode(
        "tile"
    ), change_tile_v2)
    fence.register(MetaNode(
        "tile_v2"
    ), change_tile_v2_int64)
    fence.register(MetaNode(
        "space_to_batch4d"
    ), change_to_space2depth)
    fence.register(MetaNode(
        "batch_to_space4d"
    ), change_to_depth2space)
    if version < 11:
        fence.register(MetaNode(
            "pad"
        ), change_onnx_pad_v2)
    else:
        fence.register(MetaNode(
            "pad"
        ), change_onnx_pad_v11)
    if version < 10:
        fence.register(MetaNode(
            "_resize2d"
        ), change_onnx_resize2unsample)
    else:
        raise NotImplementedError("_reisze2d not supporetd in opset_version={}".format(version))
    fence.register(MetaNode(
        "flatten"
    ), fix_flatten2d)
    fence.register(MetaNode(
        "add_bias"
    ), change_add_bias_to_add)
    fence.register(MetaNode(
        "prelu"
    ), change_prelu_slope)
    fence.register(MetaNode(
        "sample2d"
    ), change_sample2onnx)
    fence.register(MetaNode(
        "tile_v2"
    ), change_tile_v2_int64)
    return fence


node2converter = {
}


def register_node_converter(node, converter):
    # type: (str, CallableMeta) -> None
    """
    :param node:
    :param converter: assume as (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    :return:
    """
    node2converter[node] = converter


def convert2onnxnode(node, cache=None):
    # type: (ts.Node, Dict[Union[str, ts.Node], onnx.NodeProto]) -> onnx.NodeProto
    if cache is None:
        cache = {}
    if not node.name:
        return None
    if node in cache:
        return cache[node]
    op = node.op
    if op == "<param>":
        cache[node] = None
        return None
    if op not in node2converter:
        raise NotImplementedError("Not support layer {}:{} {}".format(node.op, node.name, node))

    for input in node.inputs:
        if input not in cache:
            convert2onnxnode(input, cache)

    onnxnode = node2converter[op](node, cache=cache)
    cache[node] = onnxnode
    return onnxnode


def get_tensor_stack_passes():
    return [
        "eliminate_deadend",
        "eliminate_identity",
        "eliminate_nop_dropout",
        "eliminate_nop_monotone_argmax",
        "eliminate_nop_pad",
        "eliminate_nop_transpose",
        "eliminate_unused_initializer",
        # "extract_constant_to_initializer",
        "fuse_add_bias_into_conv",
        "fuse_bn_into_conv",
        "fuse_consecutive_concats",
        "fuse_consecutive_log_softmax",
        "fuse_consecutive_reduce_unsqueeze",
        "fuse_consecutive_squeezes",
        "fuse_consecutive_transposes",
        "fuse_matmul_add_bias_into_gemm",
        "fuse_pad_into_conv",
        "fuse_transpose_into_gemm",
        "lift_lexical_references",
        "nop",
        # "split_init",
        # "split_predict",
    ]


def convert(outputs, inputs, onnx_output, version=None, rename=True):
    # type: (List[ts.Node], List[ts.Node], str, int, bool) -> None
    """
    outputs must be sorted, bottom output must be list first, no check for this
    :param outputs:
    :param inputs:
    :param onnx_output:
    :return:
    """
    if version is None:
        version = 9

    _, net_name = os.path.split(onnx_output)
    print("[INFO]: --[== Translate network...")
    # 1. zip graph, convert each nodes
    cache = {}
    outputs = _get_onnx_fence(version=version).convert(outputs, cache)
    inputs = [cache[i] for i in inputs]
    # 2. write each proto node
    # 2.1 special for inputs

    print("[INFO]: --[== Convert network...")
    # 2.2 convert each nodes
    ## copy graph
    cache = OrderedDict()
    inputs = ts.graph.clone_graph(inputs, cache)
    outputs = ts.graph.clone_graph(outputs, cache)
    ## rename each node for onnx
    cache_node_name = set()
    for i, v in cache.items():
        assert isinstance(v, ts.Node)
        v_name = v.name
        v_i = 0
        while v_name in cache_node_name:
            v_i += 1
            v_name = "{}_{}".format(v.name, v_i)
        cache_node_name.add(v_name)
        v.name = v_name

    ## now each node has one name
    node_cache = OrderedDict()
    node_cache["version"] = version

    onnx_inputs = [convert2onnxnode(i, cache=node_cache) for i in inputs]
    onnx_outputs = [convert2onnxnode(o, cache=node_cache) for o in outputs]

    ts_nodes = []
    layers = []
    for k in node_cache.keys():
        if not isinstance(k, ts.Node):
            continue
        ts_nodes.append(k)
        nodes = node_cache[k]
        if isinstance(nodes, (list, tuple)):
            layers.extend(nodes)
        else:
            layers.append(nodes)
    layers = [node for node in layers if node is not None and node.name]  # delete no computing input node

    def make_tensor_value_info(n, default_dtype=None, default_shape=None,
                               force_dtype=None, force_shape=None):
        assert isinstance(n, ts.Node)
        dtype = None
        shape = None
        if n.op == ts.Node.Const:
            value = n.get("value")
            dtype = ts.dtype.from_numpy(value.dtype)
            shape = value.shape
        else:
            shape = n.shape
            dtype = int(n.dtype) if n.has("#dtype") else None
        if dtype is None:
            dtype = default_dtype
        if shape is None:
            shape = default_shape
        if force_dtype is not None:
            dtype = force_dtype
        if force_shape is not None:
            shape = force_shape
        if dtype is None or shape is None:
            return None
        dtype_map = {
            ts.dtype.FLOAT32: onnx.TensorProto.FLOAT,
            ts.dtype.UINT8: onnx.TensorProto.UINT8,
            ts.dtype.INT8: onnx.TensorProto.INT8,
            ts.dtype.UINT16: onnx.TensorProto.UINT16,
            ts.dtype.INT16: onnx.TensorProto.INT16,
            ts.dtype.INT32: onnx.TensorProto.INT32,
            ts.dtype.INT64: onnx.TensorProto.INT64,
            ts.dtype.CHAR8: onnx.TensorProto.STRING,
            ts.dtype.BOOLEAN: onnx.TensorProto.BOOL,
            ts.dtype.FLOAT16: onnx.TensorProto.FLOAT16,
            ts.dtype.FLOAT64: onnx.TensorProto.DOUBLE,
            ts.dtype.UINT32: onnx.TensorProto.UINT32,
            ts.dtype.UINT64: onnx.TensorProto.UINT64,
            ts.dtype.COMPLEX64: onnx.TensorProto.COMPLEX64,
            ts.dtype.COMPLEX128: onnx.TensorProto.COMPLEX128,
        }
        if dtype not in dtype_map:
            return None
        return onnx.helper.make_tensor_value_info(
            n.name, dtype_map[dtype], [int(i) for i in shape]
        )

    # onnx_value_infos = [make_tensor_value_info(node) for node in ts_nodes]
    # onnx_value_infos = [i for i in onnx_value_infos if i is not None]

    print("[INFO]: --[== ONNX inputs: {}".format([node.name for node in inputs]))
    print("[INFO]: --[== ONNX outputs: {}".format([node.name for node in outputs]))

    input_infos = [make_tensor_value_info(node, force_dtype=ts.dtype.FLOAT32) for node in inputs]
    output_infos = [make_tensor_value_info(node, force_dtype=ts.dtype.FLOAT32) for node in outputs]

    print("[INFO]: --[== Convert about {} node(s). Start write files...".format(len(layers)))
    print("[INFO]: --[== Convert to onnx version: {}".format(onnx.version.version))
    # 3. output
    # 3.1 build full net
    onnx_this_graph = onnx.helper.make_graph(
        nodes=list(layers),
        name=net_name,
        inputs=input_infos,
        outputs=output_infos,
    )

    opset_imports = None
    if version is not None:
        assert isinstance(version, int)
        opset_imports=[onnx.helper.make_opsetid("", version)]

    original_model = onnx.helper.make_model(onnx_this_graph, producer_name='TenniS', opset_imports=opset_imports)
    onnx.checker.check_model(original_model)
    inferred_model = onnx.shape_inference.infer_shapes(original_model)
    onnx.checker.check_model(inferred_model)
    final_model = inferred_model
    # final_model = onnx.optimizer.optimize(final_model, get_tensor_stack_passes())
    if rename:
        final_model = tb.onnx.converter.unique_names(final_model)
    onnx.save(final_model, onnx_output)

    print("[INFO]: --[== Write files done.")

    pass


def update_blob_shape(blob_shape, shape):
    # type: (caffe.BlobShape, List[int]) -> None
    while len(blob_shape.dim):
        blob_shape.dim.pop()
    for i in shape:
        blob_shape.dim.append(i)


def update_blob(blob, data):
    # type: (caffe.BlobProto, numpy.ndarray) -> None
    data = numpy.asarray(data, dtype=numpy.float32)
    update_blob_shape(blob.shape, data.shape)
    while len(blob.data):
        blob.data.pop()
    for datum in data.reshape([-1]):
        blob.data.append(datum)


def convert_field(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto.Top
    x = convert2onnxnode(node.inputs[0], cache)
    i = int(node.get("offset"))
    return x.top(i, node.name)


# register_node_converter("_field", convert_field)


import math


def conv2d_forward(x, padding, dilation, kernel, stride):
    return int(math.floor((x + padding - (dilation * (kernel - 1) + 1)) / stride + 1))


def onnx_conv2d_forward(x, padding, dilation, kernel, stride):
    return conv2d_forward(x, padding, dilation, kernel, stride)


def conv2d_backward(y, padding, dilation, kernel, stride):
    return (y - 1) * stride + (dilation * (kernel - 1) + 1) - padding


def pooling2d_forward(x, padding, kernel, stride):
    return int(math.ceil((x + padding - kernel) / float(stride) + 1))


def pooling2d_backward(y, padding, kernel, stride):
    return (y - 1) * stride + kernel - padding


def onnx_pooling2d_forward(x, padding, kernel, stride):
    return int(math.floor((x + padding - kernel) / float(stride) + 1))


def conv2d_onnx_padding(x, padding, dilation, kernel, stride):
    # type: (int, Tuple[int], int, int, int) -> Union[Tuple[int], List[int]]
    tennis_y = conv2d_forward(x, padding[0] + padding[1], dilation, kernel, stride)
    if tennis_y == onnx_conv2d_forward(x, padding[0] + padding[0], dilation, kernel, stride):
        return [padding[0], padding[1]]

    padding_min = conv2d_backward(tennis_y, x, dilation, kernel, stride)
    padding_max = padding_min + (stride - 1)

    padding_diff = stride * 2
    padding_left = None
    padding_right = None
    for i in range(padding_min, padding_max + 1):
        if padding_min < 0:
            continue
        if i < padding[0]:
            may_padding_left = i // 2
        else:
            may_padding_left = padding[0]
        may_padding_right = i - may_padding_left
        may_padding_diff = abs(may_padding_left - padding[0]) + abs(may_padding_right - padding[1])
        if may_padding_diff < padding_diff:
            padding_left = may_padding_left
            padding_right = may_padding_right

    if padding_left is None or padding_right is None:
        raise ValueError("Conv2D can not apply positive padding with: x={}, padding={}, dilation={}, kernel={}, stride={}".format(
            x, padding, dilation, kernel, stride
        ))

    return [padding_left, padding_right]


def pooling2d_onnx_padding(x, padding, kernel, stride):
    # type: (int, Tuple[int], int, int) -> Union[Tuple[int], List[int]]
    tennis_y = pooling2d_forward(x, padding[0] + padding[1], kernel, stride)
    if tennis_y == onnx_pooling2d_forward(x, padding[0] + padding[1], kernel, stride):
        return [padding[0], padding[1]]

    padding_min = pooling2d_backward(tennis_y, x, kernel, stride)
    padding_max = padding_min + (stride - 1)

    padding_diff = stride * 2
    padding_left = None
    padding_right = None
    for i in range(padding_min, padding_max + 1):
        if padding_min < 0:
            continue
        if i < padding[0]:
            may_padding_left = i // 2
        else:
            may_padding_left = padding[0]
        may_padding_right = i - may_padding_left
        may_padding_diff = abs(may_padding_left - padding[0]) + abs(may_padding_right - padding[1])
        if may_padding_diff < padding_diff:
            padding_left = may_padding_left
            padding_right = may_padding_right

    if padding_left is None or padding_right is None:
        raise ValueError("Polling2d can not apply positive padding with: x={}, padding={}, kernel={}, stride={}".format(
            x, padding, kernel, stride
        ))

    return [padding_left, padding_right]


def convert_conv2d(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name
    w = node.inputs[1].name
    b = node.inputs[2].name if len(node.inputs) > 2 else None

    format = str(node.get("format"))
    assert format == "NCHW"

    W = node.inputs[1].get("value")

    padding = numpy.asarray(node.get("padding")).reshape([-1, 2])[-2:]
    stride = numpy.asarray(node.get("stride")).reshape([-1])[-2:]
    dilation = numpy.asarray(node.get("dilation")).reshape([-1])[-2:]
    kernel_size = W.shape[-2:]
    input_size = list(node.inputs[0].shape)[-2:]

    pad_h = conv2d_onnx_padding(input_size[0], padding[0], dilation[0], kernel_size[0], stride[0])
    pad_w = conv2d_onnx_padding(input_size[1], padding[1], dilation[1], kernel_size[1], stride[1])

    if pad_h[0] != padding[0, 0] or pad_w[0] != padding[1, 0]:
        print("[WARNING]: Layer {}:{} change padding [{}, {}] => [{}, {}]".format(
            node.op, node.name, padding[0], padding[1], pad_h, pad_w
        ))

    padding[0, :] = pad_h
    padding[1, :] = pad_w

    inputs = [x, w, b] if b is not None else [x, w]

    onp = onnx.helper.make_node("Conv", inputs, [node.name], name=node.name,
                                dilations=[dilation[0], dilation[1]],
                                kernel_shape=[kernel_size[0], kernel_size[1]],
                                pads=[pad_h[0], pad_w[0], pad_h[1], pad_w[1]],
                                strides=[stride[0], stride[1]])

    return onp


register_node_converter("conv2d", convert_conv2d)


def convert_transpose_conv2d(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name
    w = node.inputs[1].name
    b = node.inputs[2].name if len(node.inputs) > 2 else None

    format = str(node.get("format"))
    assert format == "NCHW"

    W = node.inputs[1].get("value")

    padding = numpy.asarray(node.get("padding")).reshape([-1, 2])[-2:]
    stride = numpy.asarray(node.get("stride")).reshape([-1])[-2:]
    dilation = numpy.asarray(node.get("dilation")).reshape([-1])[-2:]
    kernel_size = W.shape[-2:]
    input_size = list(node.inputs[0].shape)[-2:]

    if padding[0, 0] != 0 or \
            padding[0, 1] != 0 or \
            padding[1, 0] != 0 or \
            padding[1, 1] != 0:
        raise ValueError("tranpose conv2d not support padding {}".format(paddding))
    
    pad_h = [0, 0]
    pad_w = [0, 0]
    
    # pad_h = conv2d_onnx_padding(input_size[0], padding[0], dilation[0], kernel_size[0], stride[0])
    # pad_w = conv2d_onnx_padding(input_size[1], padding[1], dilation[1], kernel_size[1], stride[1])

    # if pad_h[0] != padding[0, 0] or pad_w[0] != padding[1, 0]:
    #    print("[WARNING]: Layer {}:{} change padding [{}, {}] => [{}, {}]".format(
    #       node.op, node.name, padding[0], padding[1], pad_h, pad_w
    #    ))

    # padding[0, :] = pad_h
    # padding[1, :] = pad_w

    inputs = [x, w, b] if b is not None else [x, w]

    onp = onnx.helper.make_node("ConvTranspose", inputs, [node.name], name=node.name,
                                dilations=[dilation[0], dilation[1]],
                                kernel_shape=[kernel_size[0], kernel_size[1]],
                                pads=[pad_h[0], pad_w[0], pad_h[1], pad_w[1]],
                                strides=[stride[0], stride[1]])

    return onp


register_node_converter("transpose_conv2d", convert_transpose_conv2d)


def convert_add_bias(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = convert2onnxnode(node.inputs[0], cache)
    cn = onnx.NodeProto("Bias", node.name, [x])
    param = cn.proto.bias_param
    blobs = cn.proto.blobs

    format = None
    dim = None
    if node.has("dim"):
        dim = int(node.get("dim"))
    if node.has("format"):
        format = str(node.get("format"))

    if dim is None:
        if format is None:
            raise ValueError("add_bias must set format and dim")
        if format == "HCHW":
            dim = 1
        elif format == "NHWC":
            dim = 3
        else:
            raise ValueError("add_bias not support format {}".format(format))

    param.axis = dim
    param.num_axes = 1
    B = node.inputs[1].get("value")

    update_blob(blobs.add(), B)

    return cn


# register_node_converter("add_bias", convert_add_bias)


def convert_pooling2d(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name

    format = str(node.get("format"))
    assert format == "NCHW"

    padding = numpy.asarray(node.get("padding")).reshape([-1, 2])[-2:]
    stride = numpy.asarray(node.get("stride")).reshape([-1])[-2:]
    ksize = numpy.asarray(node.get("ksize")).reshape([-1])[-2:]
    type = int(node.get("type"))
    input_size = list(node.inputs[0].shape)[-2:]

    pad_h = pooling2d_onnx_padding(input_size[0], padding[0], ksize[0], stride[0])
    pad_w = pooling2d_onnx_padding(input_size[1], padding[1], ksize[1], stride[1])

    if pad_h[0] != padding[0, 0] or pad_w[0] != padding[1, 0]:
        print("[WARNING]: Layer {}:{} change padding [{}, {}] => [{}, {}]".format(
            node.op, node.name, padding[0], padding[1], pad_h, pad_w
        ))

    padding[0, :] = pad_h
    padding[1, :] = pad_w

    may_type_opname = {
        0: "MaxPool",
        1: "AveragePool",
    }

    onp = onnx.helper.make_node(may_type_opname[type], [x], [node.name], name=node.name,
                                kernel_shape=[ksize[0], ksize[1]],
                                pads=[pad_h[0], pad_w[0], pad_h[1], pad_w[1]],
                                strides=[stride[0], stride[1]])

    return onp


register_node_converter("pooling2d", convert_pooling2d)


def convert_global_pooling2d(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name

    format = str(node.get("format"))
    assert format == "NCHW"

    type = int(node.get("type"))

    may_type_opname = {
        0: "GlobalMaxPool",
        1: "GlobalAveragePool",
    }

    onp = onnx.helper.make_node(may_type_opname[type], [x], [node.name], name=node.name)

    return onp


register_node_converter("global_pooling2d", convert_global_pooling2d)


def convert_const(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    value = node.get("value")

    tensor = onnx.numpy_helper.from_array(value)

    onp = onnx.helper.make_node("Constant", [], [node.name], name=node.name,
                                value=tensor)

    return onp


register_node_converter("<const>", convert_const)


def convert_fused_batch_norm(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto

    x = node.inputs[0].name
    mean = node.inputs[1].name
    var = node.inputs[2].name
    scale = node.inputs[3].name
    bias = node.inputs[4].name

    dim = int(node.try_get("dim", 1))
    epsilon = float(node.try_get("epsilon", 1e-5))

    assert dim == 1

    onp = onnx.helper.make_node("BatchNormalization", [x, scale, bias, mean, var], [node.name], name=node.name,
                                epsilon=epsilon)

    return onp


register_node_converter("fused_batch_norm", convert_fused_batch_norm)


def convert_batch_norm(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto

    x = node.inputs[0].name
    mean = node.inputs[1].name
    var = node.inputs[2].name
    scale = node.name + "_scale"
    bias = node.name + "_bias"

    dim = int(node.try_get("dim", 1))
    epsilon = float(node.try_get("epsilon", 1e-5))

    assert dim == 1

    onp = onnx.helper.make_node("BatchNormalization", [x, scale, bias, mean, var], [node.name], name=node.name,
                                epsilon=epsilon)

    mean_value = ts.inferer.infer_value(node.inputs[1])

    scale_node = onnx.helper.make_node("Constant", [], [scale], name=scale,
                                       value=onnx.numpy_helper.from_array(numpy.ones_like(mean_value)))
    bias_node = onnx.helper.make_node("Constant", [], [bias], name=bias,
                                      value=onnx.numpy_helper.from_array(numpy.zeros_like(mean_value)))

    return [scale_node, bias_node, onp]


register_node_converter("batch_norm", convert_batch_norm)


def convert_add(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name
    y = node.inputs[1].name

    onp = onnx.helper.make_node("Add", [x, y], [node.name], name=node.name)

    return onp


register_node_converter("add", convert_add)


def convert_sub(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name
    y = node.inputs[1].name

    onp = onnx.helper.make_node("Sub", [x, y], [node.name], name=node.name)

    return onp


register_node_converter("sub", convert_sub)


def convert_mul(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name
    y = node.inputs[1].name

    onp = onnx.helper.make_node("Mul", [x, y], [node.name], name=node.name)

    return onp


register_node_converter("mul", convert_mul)


def convert_div(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name
    y = node.inputs[1].name

    onp = onnx.helper.make_node("Div", [x, y], [node.name], name=node.name)

    return onp


register_node_converter("div", convert_div)


def convert_relu(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name

    onp = onnx.helper.make_node("Relu", [x], [node.name], name=node.name)

    return onp


register_node_converter("relu", convert_relu)
# register_node_converter("relu_max", convert_relu)


def convert_relu_max(node, cache):
    # type: (ts.Node, Dict[Union[str, ts.Node], onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name

    max = float(node.get("max"))

    opset_version = cache["version"]

    if opset_version >= 11:
        min_name = node.name + "_min"
        min_node = onnx.helper.make_node("Constant", [], [min_name], name=min_name,
                                          value=onnx.numpy_helper.from_array(numpy.asarray(0, dtype=numpy.float32)))
        max_name = node.name + "_max"
        max_node = onnx.helper.make_node("Constant", [], [max_name], name=max_name,
                                         value=onnx.numpy_helper.from_array(numpy.asarray(max, dtype=numpy.float32)))
        onp = onnx.helper.make_node("Clip", [x, min_name, max_name], [node.name], name=node.name)
        return min_node, max_node, onp
    else:
        onp = onnx.helper.make_node("Clip", [x], [node.name], name=node.name,
                                    min=0.0, max=max)
        return onp


register_node_converter("relu_max", convert_relu_max)


def convert_sigmoid(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name

    onp = onnx.helper.make_node("Sigmoid", [x], [node.name], name=node.name)

    return onp


register_node_converter("sigmoid", convert_sigmoid)


def convert_hard_sigmoid(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name

    onp = onnx.helper.make_node("HardSigmoid", [x], [node.name], name=node.name)

    return onp


register_node_converter("hard_sigmoid", convert_hard_sigmoid)


def convert_tanh(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name

    onp = onnx.helper.make_node("Tanh", [x], [node.name], name=node.name)

    return onp


register_node_converter("tanh", convert_tanh)


def convert_abs(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name

    onp = onnx.helper.make_node("Abs", [x], [node.name], name=node.name)

    return onp


register_node_converter("abs", convert_abs)


def convert_gemm(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    a = node.inputs[0].name
    b = node.inputs[1].name
    c = node.inputs[2].name if len(node.inputs) > 2 else None

    alpha = float(node.get("alpha"))
    beta = float(node.get("beta"))
    transA = bool(node.get("transA"))
    transB = bool(node.get("transB"))

    inputs = [a, b, c] if c is not None else [a, b]

    onp = onnx.helper.make_node("Gemm", inputs, [node.name], name=node.name,
                                alpha=alpha, beta=beta,
                                transA=transA, transB=transB)

    return onp


register_node_converter("gemm", convert_gemm)


def convert_inner_prod(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto

    x = node.inputs[0].name
    w = node.inputs[1].name
    b = node.inputs[2].name if len(node.inputs) > 2 else None

    zero_b = True
    if len(node.inputs) > 2:
        zero_b = numpy.all(ts.inferer.infer_value(node.inputs[2]) == 0)

    transpose = bool(node.try_get("transpose", False))

    inputs = [x, w, b] if b is not None else [x, w]
    alpha = 1.0
    beta = 1.0 if b is not None else 0.0
    transB = transpose

    if zero_b:
        beta = 0.0

    onp = onnx.helper.make_node("Gemm", inputs, [node.name], name=node.name,
                                alpha=alpha, beta=beta,
                                transB=transB)

    return onp


register_node_converter("inner_prod", convert_inner_prod)


def convert_mat_mul(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto

    x = node.inputs[0].name
    w = node.inputs[1].name

    onp = onnx.helper.make_node("MatMul", [x, w], [node.name], name=node.name)

    return onp


register_node_converter("mat_mul", convert_mat_mul)
register_node_converter("matmul", convert_mat_mul)


def convert_concat(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    inputs = [i.name for i in node.inputs]

    dim = int(node.get("dim"))

    onp = onnx.helper.make_node("Concat", inputs, [node.name], name=node.name,
                                axis=dim)

    return onp


register_node_converter("concat", convert_concat)


def convert_neg(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = convert2onnxnode(node.inputs[0], cache)
    cn = onnx.NodeProto("Power", node.name, [x])
    param = cn.proto.power_param
    blobs = cn.proto.blobs
    # Use Power layer: power = 1, scale = -1.0, shift = 0
    param.power = 1
    param.scale = -1
    param.shift = 0

    return cn


# register_node_converter("neg", convert_neg)


def convert_transpose(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name

    permute = list(node.get("permute"))

    onp = onnx.helper.make_node("Transpose", [x], [node.name], name=node.name,
                                perm=permute)

    return onp


register_node_converter("_transpose", convert_transpose)


def convert_group_conv2d(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name
    w = node.inputs[1].name
    b = node.inputs[2].name if len(node.inputs) > 2 else None

    format = str(node.get("format"))
    assert format == "NCHW"

    group = int(node.try_get("group", 1))

    W = node.inputs[1].get("value")

    padding = numpy.asarray(node.get("padding")).reshape([-1, 2])[-2:]
    stride = numpy.asarray(node.get("stride")).reshape([-1])[-2:]
    dilation = numpy.asarray(node.get("dilation")).reshape([-1])[-2:]
    kernel_size = W.shape[-2:]
    input_size = list(node.inputs[0].shape)[-2:]

    pad_h = conv2d_onnx_padding(input_size[0], padding[0], dilation[0], kernel_size[0], stride[0])
    pad_w = conv2d_onnx_padding(input_size[1], padding[1], dilation[1], kernel_size[1], stride[1])

    if pad_h[0] != padding[0, 0] or pad_w[0] != padding[1, 0]:
        print("[WARNING]: Layer {}:{} change padding [{}, {}] => [{}, {}]".format(
            node.op, node.name, padding[0], padding[1], pad_h, pad_w
        ))

    padding[0, :] = pad_h
    padding[1, :] = pad_w

    inputs = [x, w, b] if b is not None else [x, w]

    onp = onnx.helper.make_node("Conv", inputs, [node.name], name=node.name,
                                dilations=[dilation[0], dilation[1]],
                                kernel_shape=[kernel_size[0], kernel_size[1]],
                                pads=[pad_h[0], pad_w[0], pad_h[1], pad_w[1]],
                                strides=[stride[0], stride[1]],
                                group=group)

    return onp


register_node_converter("group_conv2d", convert_group_conv2d)


def convert_depthwise_conv2d_nnie(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = convert2onnxnode(node.inputs[0], cache)
    cn = onnx.NodeProto("DepthwiseConv", node.name, [x])
    param = cn.proto.convolution_param
    blobs = cn.proto.blobs

    format = str(node.get("format"))
    assert format == "NCHW"

    W = node.inputs[1].get("value")

    number_filters = W.shape[0]
    input_channels = W.shape[1]
    output_channels = number_filters * input_channels

    W = numpy.transpose(W, (1, 0, 2, 3))    # change number filter to dim 1
    W = numpy.reshape(W, [output_channels, 1, W.shape[2], W.shape[3]])

    update_blob(blobs.add(), W)

    param.num_output = output_channels
    # param.group = input_channels  # no group parameter for depthwise

    padding = numpy.asarray(node.get("padding")).reshape([-1, 2])[-2:]
    stride = numpy.asarray(node.get("stride")).reshape([-1])[-2:]
    dilation = numpy.asarray(node.get("dilation")).reshape([-1])[-2:]
    kernel_size = W.shape[-2:]
    input_size = list(node.inputs[0].shape)[-2:]

    pad_h = conv2d_same_padding(input_size[0], padding[0], dilation[0], kernel_size[0], stride[0])
    pad_w = conv2d_same_padding(input_size[1], padding[1], dilation[1], kernel_size[1], stride[1])

    if pad_h[0] != padding[0, 0] or pad_w[0] != padding[1, 0]:
        print("[WARNING]: Layer {}:{} change padding [{}, {}] => [{}, {}]".format(
            node.op, node.name, padding[0], padding[1], pad_h, pad_w
        ))

    padding[0, :] = pad_h
    padding[1, :] = pad_w

    if kernel_size[0] == kernel_size[1]:
        param.kernel_size.extend(kernel_size[-1:])
    else:
        param.kernel_size.extend(kernel_size[-2:])

    if dilation[0] == dilation[1]:
        param.dilation.extend(dilation[-1:])
    else:
        param.dilation.extend(dilation[-2:])

    if stride[0] == stride[1]:
        param.stride.extend(stride[-1:])
    else:
        param.stride.extend(stride[-2:])

    assert padding[0, 0] == padding[0, 1]
    assert padding[1, 0] == padding[1, 1]

    if padding[0, 0] == padding[1, 0]:
        param.pad.extend([padding[0, 0]])
    else:
        param.pad.extend([padding[0, 0], padding[1, 0]])

    if len(node.inputs) > 2:
        B = node.inputs[2].get("value")
        update_blob(blobs.add(), B)

        param.bias_term = True
    else:
        param.bias_term = False

    return cn


def convert_softmax(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name
    shape = node.shape

    dim = int(node.get("dim"))

    # Notice: dim and content has been freezed, so just convert it
    onp = onnx.helper.make_node("Softmax", [x], [node.name], name=node.name,
                                axis=dim)

    return onp


register_node_converter("softmax", convert_softmax)


def convert_reshape_v2(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name
    shape = node.inputs[1].name

    onp = onnx.helper.make_node("Reshape", [x, shape], [node.name], name=node.name)

    return onp


register_node_converter("onnx::reshape", convert_reshape_v2)


def convert_tile_v2(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name
    shape = node.inputs[1].name

    onp = onnx.helper.make_node("Tile", [x, shape], [node.name], name=node.name)

    return onp


register_node_converter("tile_v2", convert_tile_v2)
register_node_converter("onnx::tile", convert_tile_v2)


def convert_gather(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name
    indices = node.inputs[1].name

    axis = int(node.get("axis"))

    onp = onnx.helper.make_node("Gather", [x, indices], [node.name], name=node.name,
                                axis=axis)

    return onp


register_node_converter("gather", convert_gather)


def convert_flatten(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name

    dim = int(node.try_get("dim", 1))

    onp = onnx.helper.make_node("Flatten", [x], [node.name], name=node.name,
                                axis=dim)

    return onp


register_node_converter("flatten", convert_flatten)


def convert_sub(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = convert2onnxnode(node.inputs[0], cache)
    y = convert2onnxnode(node.inputs[1], cache)
    cn = onnx.NodeProto("Eltwise", node.name, [x, y])
    param = cn.proto.eltwise_param
    blobs = cn.proto.blobs

    # 0-PROD, 1-SUM, 2-MAX
    param.operation = 1
    param.coeff.extend([1, -1])

    return cn


# register_node_converter("sub", convert_sub)


def convert_mul(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = convert2onnxnode(node.inputs[0], cache)
    y = convert2onnxnode(node.inputs[1], cache)
    cn = onnx.NodeProto("Eltwise", node.name, [x, y])
    param = cn.proto.eltwise_param
    blobs = cn.proto.blobs

    # 0-PROD, 1-SUM, 2-MAX
    param.operation = 0

    return cn


# register_node_converter("mul", convert_mul)


def convert_threshold(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = convert2onnxnode(node.inputs[0], cache)
    cn = onnx.NodeProto("Threshold", node.name, [x])
    param = cn.proto.threshold_param
    blobs = cn.proto.blobs

    param.threshold = float(node.get("threshold"))

    return cn


# register_node_converter("threshold", convert_threshold)


def convert_copy(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name

    onp = onnx.helper.make_node("Identity", [x], [node.name], name=node.name,)

    return onp


register_node_converter("_copy", convert_copy)


def convert_onnx_pad_v11(node, cache):
    # type: (ts.Node, Dict[Union[str, ts.Node], onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name
    pads = node.inputs[1].name

    assert cache["version"] >= 11

    onp = onnx.helper.make_node("Pad", [x, pads], [node.name], name=node.name)

    return onp


register_node_converter("onnx::pad-11", convert_onnx_pad_v11)


def convert_onnx_pad_v2(node, cache):
    # type: (ts.Node, Dict[Union[str, ts.Node], onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name

    assert 11 > cache["version"] >= 2

    padding = node.get("padding")

    # notice: fence already change to onnx format
    # padding = numpy.transpose(numpy.asarray(padding).reshape([-1, 2])).reshape([-1])

    onp = onnx.helper.make_node("Pad", [x], [node.name], name=node.name,
                                pads=list(numpy.asarray(padding, dtype=numpy.int32)))

    return onp


register_node_converter("onnx::pad-2", convert_onnx_pad_v2)


def convert_space_to_depth4d(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name

    # padding = numpy.asarray(node.get("padding")).reshape([2, 2])
    block_shape = numpy.asarray(node.get("block_shape")).reshape([2])

    assert block_shape[0] == block_shape[1]

    onp = onnx.helper.make_node("SpaceToDepth", [x], [node.name], name=node.name,
                                blocksize=block_shape[0])

    return onp


register_node_converter("space_to_depth4d", convert_space_to_depth4d)


def convert_depth_to_space4d(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name

    # padding = numpy.asarray(node.get("padding")).reshape([2, 2])
    block_shape = numpy.asarray(node.get("block_shape")).reshape([2])

    assert block_shape[0] == block_shape[1]

    onp = onnx.helper.make_node("DepthToSpace", [x], [node.name], name=node.name,
                                blocksize=block_shape[0])

    return onp


register_node_converter("depth_to_space4d", convert_depth_to_space4d)


def convert_squeeze(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name

    axes = list(node.get("axes"))

    onp = onnx.helper.make_node("Squeeze", [x], [node.name], name=node.name,
                                axes=axes)

    return onp


register_node_converter("squeeze", convert_squeeze)


def convert_leaky_relu(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name

    scale = float(node.get("scale"))

    onp = onnx.helper.make_node("LeakyRelu", [x], [node.name], name=node.name,
                                alpha=scale)

    return onp


register_node_converter("leaky_relu", convert_leaky_relu)


def convert_onnx_unsample(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name
    scales = node.inputs[1].name

    type = int(node.get("type"))

    type_map = {
        0: "bilinear",
        1: "bilinear",  # cubic
        2: "nearest",   # nearest
        3: "nearest",   # hard
    }

    onp = onnx.helper.make_node("Upsample", [x, scales], [node.name], name=node.name,
                                mode=type_map[type])

    return onp


register_node_converter("onnx::unsample", convert_onnx_unsample)


def convert_prelu(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name
    slope = node.inputs[1].name

    dim = int(node.get("dim"))
    assert dim == 1

    onp = onnx.helper.make_node("PRelu", [x, slope], [node.name], name=node.name)

    return onp


register_node_converter("onnx::prelu", convert_prelu)


def convert_lstm(node, cache):
    # type: (ts.Node, Dict[ts.Node, onnx.NodeProto]) -> onnx.NodeProto
    x = node.inputs[0].name
    w = node.inputs[1].name
    r = node.inputs[2].name
    b = node.inputs[3].name
    h = node.inputs[4].name
    c = node.inputs[5].name

    direction = str(node.get("direction"))
    hidden_size = int(node.get("hidden_size"))
    
    inputs = [x, w, r, b, '', h, c]

    onp = onnx.helper.make_node("LSTM", inputs, [node.name], name=node.name,
                                direction=direction,
                                hidden_size=hidden_size)

    return onp


register_node_converter("LSTM", convert_lstm)
