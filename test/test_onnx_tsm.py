import sys
import time
import cv2
import numpy as np
tennis = '/home/kier/git/TensorStack/python'
sys.path.append(tennis)

from tennis.backend import api as api


def inference(path, bench, filter):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img[:, :, :]
    if img.ndim != 3 or img.shape[2] != 3:
        raise TypeError('Only RGB images are supported.')

    tensor = api.Tensor(img, api.FLOAT32, (1, img.shape[0], img.shape[1], img.shape[2]))
    tensor = filter.run(tensor)
    bench.input(0, tensor)
    start = time.time()
    bench.run()
    infer_time = time.time() - start
    feature = bench.output(0).cast(api.FLOAT32).numpy
    # print(infer_time)
    # print(feature)
    return feature


class RKNN2(api.Operator):
    def __init__(self):
        super(RKNN2, self).__init__()
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


api.RegisterOperator(RKNN2, "cpu", "rknn2")


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


def export(src, dst):
    import tennis_rknn as rknn

    exporter = rknn.exporter.RKNNExporter()
    exporter.load(src)

    exporter.export_tsm_with_rknn(dst)


def test():
    img_path = "/home/kier/Documents/rknn-toolkit-v0.9.3/rknn_example/onnx/arcface_112x112/debug/00000.jpg"
    origin_model_path = "/home/kier/Documents/rknn-toolkit-v0.9.3/rknn_example/onnx/arcface_112x112/resnet50_iter_118000.tsm"
    model_path = "debug/test.tsm"

    export(origin_model_path, model_path)

    device = api.Device("cpu", 0)

    bench = api.Workbench(device=device)
    bench.setup_context()
    bench.setup_device()
    bench.set_computing_thread_number(8)

    filter = api.ImageFilter(device=device)
    filter.to_float()
    filter.sub_mean([0.0, 0.0, 0.0])
    filter.div_std([255.0, 255.0, 255.0])
    filter.to_chw()

    bench.setup(bench.compile(api.Module.Load(model_path)))

    feature = inference(img_path, bench, filter)

    gt = np.load("debug.tsm.npy")

    filter.dispose()
    bench.dispose()

    cos = compare(gt, feature)
    avg, max = diff(gt, feature)
    print("avg={}, max={}, cos={}".format(avg, max, cos))



if __name__ == '__main__':
    test()
