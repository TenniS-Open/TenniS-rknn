from tennisfence.spliter import GraphSpliter
from tennisfence.metanode import *

from typing import Optional


def _get_shape(node):
    # type: (ts.Node) -> Optional[List[int]]
    if node.has("#shape"):
        return node.shape
    if node.op == ts.Node.Const:
        return node.get("value").shape
    return None


def if_no_broadcast_reduce(op):
    # type: (str) -> CallableMeta
    def checker(node):
        # type: (ts.Node) -> bool
        if node.op != op:
            return False
        lhs = node.inputs[0]
        rhs = node.inputs[1]
        lhs_shape = _get_shape(lhs)
        rhs_shape = _get_shape(rhs)
        if lhs_shape is None or rhs_shape is None:
            return False
        if len(lhs_shape) != len(rhs_shape):
            return False
        for i, j in zip(lhs_shape, rhs_shape):
            if i != j:
                return False
        return True
    return checker


def if_non_batch_transpose(node):
    # type: (ts.Node) -> bool
    if node.op != "_transpose":
        return False
    permute = list(node.get("permute"))
    if len(permute) < 1:
        return False
    if permute[0] != 0:
        return False
    return True


ALLOWED_DOWNSAMPLE = True


def upsample_resize(node):
    # type: (ts.Node) -> bool
    if node.op != "_resize2d":
        return False

    t_size = ts.inferer.infer_value(node.inputs[1])
    if t_size is None:
        return False

    x_size = node.inputs[0].shape
    y_size = node.shape
    if x_size is None or y_size is None:
        return False

    if len(x_size) != len(y_size) or len(y_size) != len(t_size):
        return False

    if ALLOWED_DOWNSAMPLE:
        return True

    for x, y, t in zip(x_size, y_size, t_size):
        if t <= 0:
            continue
        if x <= 0 or y <= 0:
            return False
        if x > y:
            return False
        continue

    return True


def infered_flatten(node):
    # type: (ts.Node) -> bool
    if node.op != "flatten":
        return False

    shape = node.shape
    if shape is None:
        return False

    return numpy.all(shape[1:] > 0)


def get_spliter():
    # type: () -> GraphSpliter
    gs = GraphSpliter(only_max_graph_out=True, single_input=True, single_output=False,
                      log_end_nodes=True)
    gs.route(ts.Node.Const)
    gs.support(MetaGraph([
        ts.Node.Const,
        (MetaNode({
            "#op": "conv2d",
            "#shape": GT([None, 0, 0, 0])
        }), {1: -1})
    ]))
    gs.support(MetaGraph([
        ts.Node.Const,
        (MetaNode({
            "#op": "transpose_conv2d",
            "#shape": GT([None, 0, 0, 0])
        }), {1: -1})
    ]))
    gs.support(MetaGraph([
        ts.Node.Const,
        ("add_bias", {1: -1})
    ]))
    gs.support(MetaGraph([
        ts.Node.Const,
        ("depthwise_conv2d", {1: -1})
    ]))
    gs.support("pooling2d")
    gs.support(MetaGraph([
        ts.Node.Const,
        ts.Node.Const,
        ts.Node.Const,
        ts.Node.Const,
        ("fused_batch_norm", {1: -1, 2: -2, 3: -3, 4: -4})
    ]))
    gs.support(MetaGraph([
        ts.Node.Const,
        ts.Node.Const,
        ("batch_norm", {1: -1, 2: -2})
    ]))
    gs.support(MetaGraph([
        ts.Node.Const,
        ts.Node.Const,
        ("batch_scale", {1: -1, 2: -2})
    ]))
    gs.support("add")
    # gs.support(if_no_broadcast_reduce("add"))
    gs.support("relu")
    gs.support("inner_prod")
    # gs.route(MetaNode("_reshape", shape=HasShape(4) & (EQ([0, None, None, None]) | EQ([1, None, None, None]))))
    gs.support(MetaGraph([
        {"#op": "concat",
         "dim": NE(0)}
    ]))
    # gs.support("sub")
    # gs.support(if_no_broadcast_reduce("sub"))
    # gs.support(MetaGraph([
    #     {"#op": ts.Node.Const, "value": EQ(0)},
    #     ({"#op": "sub", "#shape": HasShape(4)}, {0: -1})
    # ]))
    # gs.support(MetaNode({
    #     "#op": "_transpose",
    #     "permute": EQ([0, 2, 3, 1])
    # }))
    gs.support("_transpose")
    # gs.support(if_non_batch_transpose)
    # gs.support(lambda x: x.op[:6] == "caffe:")
    gs.support(MetaNode({
        "#op": "softmax",
    }))
    # gs.support(MetaNode({
    #     "#op": "relu_max",
    #     "max": 6,
    #     "#shape": GT([None, 0, 0, 0])
    # }))
    gs.route("_copy")
    gs.support("mul")
    gs.route("_reshape")
    gs.route("_reshape_v2")
    gs.support("gemm")
    # gs.support("_dimshuffle")
    gs.support("global_pooling2d")
    gs.support("sigmoid")
    gs.support("hard_sigmoid")
    gs.support(MetaGraph([
        ts.Node.Const,
        ({"#op": "broadcast"}, {1: -1})
    ]))
    gs.support("tanh")
    gs.support("abs")
    # gs.support("tile")    # rknn not support tile
    gs.support("relu_max")
    # gs.support("space_to_batch4d")
    # gs.support("batch_to_space4d")
    gs.support("squeeze")
    gs.support("leaky_relu")
    gs.support(upsample_resize)
    # gs.route("flatten")
    gs.route(infered_flatten)
    gs.support("prelu")
    gs.support("sample2d")

    gs.support(if_no_broadcast_reduce("mul"))
    # gs.support(if_no_broadcast_reduce("div"))
    gs.support(MetaGraph([
        {"#op": ts.Node.Const, "value": HasShape()},
        ({"#op": "div", "#shape": HasShape(4)}, {1: -1})
    ]))
    gs.support(MetaGraph([
        ts.Node.Const,
        ("tile_v2", {1: -1})
    ]))
    gs.support("LSTM")
    gs.support("matmul")

    return gs
