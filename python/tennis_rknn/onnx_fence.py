from tennisfence.fence import Fence
from tennisfence.metanode import *

from typing import Optional


def fuse_flatten_ip_bias_reshape(node):
    # type: (ts.Node) -> Optional[ts.Node]
    reshape = node
    add_bias = reshape.inputs[0]
    inner_prod = add_bias.inputs[0]
    flatten = inner_prod.inputs[0]
    x = flatten.inputs[0]
    W = inner_prod.inputs[1]
    B = add_bias.inputs[1]

    if flatten.has("dim") and int(flatten.get("dim")) != 1:
        return None

    shape = list(reshape.get("shape"))
    if len(shape) != 4 or shape[2] != 1 or shape[3] != 1:
        return None

    caffe_inner_prod = ts.graph.clone_bubble(inner_prod)
    caffe_inner_prod.name = node.name
    caffe_inner_prod.op = "caffe:inner_prod"

    ts.Node.Link(caffe_inner_prod, [x, W, B])

    return caffe_inner_prod


def fuse_flatten_ip_reshape(node):
    # type: (ts.Node) -> Optional[ts.Node]
    reshape = node
    inner_prod = reshape.inputs[0]
    flatten = inner_prod.inputs[0]
    x = flatten.inputs[0]
    W = inner_prod.inputs[1]

    if flatten.has("dim") and int(flatten.get("dim")) != 1:
        return None

    shape = list(reshape.get("shape"))
    if len(shape) != 4 or shape[2] != 1 or shape[3] != 1:
        return None

    caffe_inner_prod = ts.graph.clone_bubble(inner_prod)
    caffe_inner_prod.name = node.name
    caffe_inner_prod.op = "caffe:inner_prod"

    ts.Node.Link(caffe_inner_prod, [x, W])

    return caffe_inner_prod


def fuse_softmax(node):
    # type: (ts.Node) -> Optional[ts.Node]
    name = node.name
    exp = node.inputs[0]
    reduce_sum = node.inputs[1]
    x = exp.inputs[0]
    assert isinstance(reduce_sum, ts.Node)

    dims = reduce_sum.get("dims")
    keep_dims = reduce_sum.try_get("keep_dims", True)

    if not keep_dims:
        return None

    dims = numpy.asarray(dims).reshape([-1])
    if len(dims) > 1:
        return None

    dim = dims[0]

    softmax = ts.zoo.softmax(name=name, x=x, dim=dim, smooth=False)
    if node.has("#shape"):
        softmax.shape = node.shape
    if node.has("#dtype"):
        softmax.dtype = node.dtype

    return softmax


def convert_reshape_v2_to_v1(node):
    # type: (ts.Node) -> Optional[ts.Node]
    name = node.name
    reshape_v2 = node
    x = reshape_v2.inputs[0]
    shape = reshape_v2.inputs[1]
    assert isinstance(shape, ts.Node)

    if shape.op == ts.Node.Const:
        shape = shape.get("value")
    elif shape.has("#value"):
        shape = shape.get("#value")
    else:
        return None

    neg_count = 0
    for i in shape:
        if i < 0:
            neg_count += 1

    if neg_count > 1:
        return None

    reshape = ts.zoo.reshape(name, x, shape)
    if node.has("#shape"):
        reshape.shape = node.shape
    if node.has("#dtype"):
        reshape.dtype = node.dtype

    return reshape


def convert_sample2d_v2_to_v1(node):
    # type: (ts.Node) -> Optional[ts.Node]
    name = node.name
    sample2d_v2 = node
    x = sample2d_v2.inputs[0]
    scales = sample2d_v2.inputs[1]
    assert isinstance(scales, ts.Node)

    if scales.op == ts.Node.Const:
        scales = scales.get("value")
    elif scales.has("#value"):
        scales = scales.get("#value")
    else:
        return None

    if not numpy.all(scales[:-2] == 1):
        return None

    if scales[-1] != scales[-2]:
        return None

    scale = scales[-1]

    sample2d = ts.zoo.sample2d(name, x, scale=scale, type=node.get("type"))
    if node.has("#shape"):
        sample2d.shape = node.shape
    if node.has("#dtype"):
        sample2d.dtype = node.dtype

    return sample2d


def convert_reshape_flatten(node):
    # type: (ts.Node) -> Optional[ts.Node]
    name = node.name
    x = node.inputs[0]
    reshape = node

    shape = list(reshape.get("shape"))

    if len(shape) != 2:
        return None

    x_shape = list(x.shape)

    if len(x_shape) < 2:
        return None

    if shape[0] != 0 and x_shape[0] != shape[0]:
        return None

    if shape[1] >= 0:
        return None

    flatten = ts.zoo.flatten(name, x)
    if node.has("#shape"):
        flatten.shape = node.shape
    if node.has("#dtype"):
        flatten.dtype = node.dtype

    return flatten


def convert_flatten_concat(node):
    # type: (ts.Node) -> Optional[ts.Node]
    name = node.name

    is_4d = MetaNode({"#shape": HasShape(4)})

    concat = node
    flatten_x = concat.inputs
    for n in flatten_x:
        if n.op != "flatten":
            return None
    x = [n.inputs[0] for n in flatten_x]
    for n in x:
        if not is_4d(n):
            return None

    concat_4d_name = name + "_4d_concat"

    # convert sub graph to reshape([0, -1, 1, 1]), concat, flatten
    reshape_x = []
    for i, n in enumerate(x):
        reshape_x.append(ts.zoo.reshape(name="%s_%d" % (concat_4d_name, i), x=n, shape=[0, -1, 1, 1]))

    concat_4d = ts.zoo.concat(concat_4d_name, reshape_x, dim=1)

    flatten_concat_4d = ts.zoo.flatten(name, concat_4d)

    # update each shape
    if node.has("#dtype"):
        dtype = node.dtype
        for n in reshape_x:
            n.dtype = dtype
        concat_4d.dtype = dtype
        flatten_concat_4d.dtype = dtype

    flatten_concat_4d.shape = node.shape
    concat_4d.shape = numpy.concatenate([node.shape, [1, 1]])
    for i, rx in enumerate(reshape_x):
        rx.shape = numpy.concatenate([flatten_x[i].shape, [1, 1]])

    return flatten_concat_4d


def convert_flatten_gemm_to_inner_prod(node):
    # type: (ts.Node) -> Optional[ts.Node]
    name = node.name

    gemm = node
    flatten = gemm.inputs[0]
    x = flatten.inputs[0]
    W = gemm.inputs[1]
    B = gemm.inputs[2]

    M, N = tuple(gemm.shape)

    alpha = float(gemm.get("alpha"))
    beta = float(gemm.get("beta"))
    transA = bool(gemm.get("transA"))
    transB = bool(gemm.get("transB"))

    if abs(alpha - 1) > 1e-6:
        return None

    if transA:
        return None

    if abs(beta) > 1e-6:
        B_value = numpy.asarray(B.get("value"))
        if B_value.shape == (N, ):
            pass
        elif B_value.shape == (1, ) or B_value.shape == ():
            tmp = float(B_value)
            B_value = numpy.zeros((N, ), dtype=numpy.float32)
            B_value[:] = tmp
            B = ts.menu.data(name=B.name + "_broadcast", value=B_value)
        else:
            return None
    else:
        B = None

    inner_prod_name = name + "_ip"
    transpose = transB
    if B is None:
        inner_prod = ts.menu.op(name=inner_prod_name, op_name="caffe:inner_prod", inputs=[x, W])
    else:
        inner_prod = ts.menu.op(name=inner_prod_name, op_name="caffe:inner_prod", inputs=[x, W, B])
    inner_prod.set("transpose", transpose, numpy.bool)

    flatten_inner_prod = ts.zoo.flatten(name, inner_prod)

    # update each shape
    if node.has("#dtype"):
        dtype = node.dtype
        flatten_inner_prod.dtype = dtype
        inner_prod.dtype = dtype

    inner_prod.shape = numpy.concatenate([gemm.shape, [1, 1]])
    flatten_inner_prod.shape = node.shape

    return flatten_inner_prod


def const_transpose(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]
    x_value = ts.inferer.infer_value(x)
    if x_value is None:
        return None

    permute = list(node.get("permute"))

    t_value = numpy.transpose(x_value, permute)

    t_node = ts.menu.data(name=node.name, value=t_value)
    if node.has("#device"):
        t_node.set("#device", node.get("#device"))

    if node.has("#dtype"):
        dtype = node.dtype
        t_node.dtype = dtype

    t_node.shape = t_value.shape

    return t_node


def const_shape(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]
    shape = ts.inferer.infer_value(node)
    if shape is None:
        return None

    shape = numpy.asarray(shape, dtype=numpy.int32)

    shape_node = ts.menu.data(node.name, shape, device=ts.device.CPU)
    shape_node.shape = shape.shape
    shape_node.dtype = ts.dtype.from_numpy(shape.dtype)

    return shape_node


def const_cast(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]
    x_value = ts.inferer.infer_value(x)
    if x_value is None:
        return None

    dtype = int(node.get("dtype"))

    x_cast_value = numpy.asarray(x_value, dtype=ts.dtype.to_numpy(dtype))

    cast_node = ts.menu.data(node.name, x_cast_value, device=ts.device.CPU)
    cast_node.shape = x_value.shape
    cast_node.dtype = dtype

    return cast_node


def check_transpose_reshape(may_reshape, dim):
    # type: (ts.Node, int) -> bool
    if may_reshape.op not in {"_reshape"}:
        return False
    shape = list(may_reshape.get("shape"))
    if dim < 0:
        dim += len(shape)
    for i, v in enumerate(shape):
        if i == dim:
            continue
        if dim < 0:
            return False
    may_transpose = may_reshape.inputs[0]
    assert isinstance(may_transpose, ts.Node)
    if may_transpose.op != "_transpose":
        return False
    transpose_shape = may_transpose.shape
    reshape_shape = may_reshape.shape
    if reshape_shape is None or transpose_shape is None:
        return False
    if len(transpose_shape) < len(reshape_shape):
        return False
    transpose_shape = numpy.asarray(transpose_shape)
    reshape_shape = numpy.asarray(reshape_shape)
    left = dim
    right = len(shape) - dim
    if not numpy.all(transpose_shape[:left] == reshape_shape[:left]):
        return False
    if not numpy.all(transpose_shape[-right:] == reshape_shape[-right:]):
        return False

    # check if transpose move
    if shape[dim] < 0:
        return False

    return True


def broadcast_scalar_lhs(node):
    # type: (ts.Node) -> Optional[ts.Node]
    shape = node.get("#shape")
    lhs = node.inputs[0]
    rhs = node.inputs[1]
    rhs_data = ts.inferer.infer_value(rhs)

    rhs_data = numpy.broadcast_to(rhs_data, shape)
    new_node = ts.graph.clone_bubble(node)
    rhs_node = ts.menu.data(name=rhs.name, value=rhs_data)
    ts.Node.Link(new_node, [lhs, rhs_node])

    return new_node


def convert_transpose_reshape_flatten(node):
    # type: (ts.Node) -> Optional[ts.Node]
    if node.op != "concat":
        return None
    # TODO: not work now
    return None


def get_fence():
    # type: () -> Fence
    fence = Fence()
    #
    # fence.register(MetaGraph([
    #     ts.Node.Const,
    #     ts.Node.Const,
    #     {"#op": "flatten"},
    #     ({"#op": "inner_prod"}, [-1, ABS(0)]),
    #     ({"#op": "add_bias"}, [-1, ABS(1)]),
    #     ({"#op": "_reshape"}, -1),
    # ]), fuse_flatten_ip_bias_reshape)
    #
    # fence.register(MetaGraph([
    #     ts.Node.Const,
    #     {"#op": "flatten"},
    #     ({"#op": "inner_prod"}, [-1, ABS(0)]),
    #     ({"#op": "_reshape"}, -1,),
    # ]), fuse_flatten_ip_reshape)
    #
    fence.register(MetaGraph([
        MetaNode(),
        ({"#op": "exp"}, -1),
        ({"#op": "reduce_sum"}, -1),
        ({"#op": "div"}, (-2, -1)),
    ]), fuse_softmax)
    #
    fence.register(MetaNode({
            "#op": "_reshape_v2",
            "#shape": HasSet,
            "#dtype": NE(0),
        }), convert_reshape_v2_to_v1)
    #
    # fence.register(MetaGraph([
    #     {"#shape": HasSet},
    #     ("_reshape", -1)
    # ]), convert_reshape_flatten)
    #
    # fence.register(MetaGraph([
    #     {"#op": "concat",
    #      "dim": EQ(1) | EQ(-3)},
    # ]), convert_flatten_concat)
    #
    # fence.register(MetaGraph([
    #     ts.Node.Const,
    #     ts.Node.Const,
    #     {"#shape": HasShape(4)},
    #     ({"#op": "flatten"}, -1),
    #     ({"#op": "gemm"}, [-1, ABS(0), ABS(1)]),
    # ]), convert_flatten_gemm_to_inner_prod)
    fence.register(MetaNode("_transpose"),
                   const_transpose)
    fence.register(MetaNode("_shape"),
                   const_shape)
    fence.register(MetaNode("_cast"),
                   const_cast)
    fence.register(MetaNode("concat"),
                   convert_transpose_reshape_flatten)

    fence.register(MetaGraph([
       {"#op": ts.Node.Const},
       ({"#op": "sample2d_v2", "#shape": HasShape(4)}, {1: -1})
    ]), convert_sample2d_v2_to_v1)
    # fence.register(MetaGraph([
    #     {"#op": ts.Node.Const, "value": HasShape()},
    #     ({"#op": "div", "#shape": HasSet}, {1: -1})
    # ]), broadcast_scalar_lhs)
    # fence.register(MetaGraph([
    #     {"#op": ts.Node.Const, "value": HasShape()},
    #     ({"#op": "add", "#shape": HasSet}, {1: -1})
    # ]), broadcast_scalar_lhs)

    return fence


def throw_non_back(node):
    # type: (ts.Node) -> Optional[ts.Node]
    raise Exception("No registered converter for {}".format(node.op))


def back_fence():
    # type: () -> Fence
    fence = Fence()

    fence.register(
        lambda x: x.op[:6] == "onnx::",
        throw_non_back,
        -1
    )

    return fence

