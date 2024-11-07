import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

onnx_model = onnx.load("tiny.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("model_tf")

converter = tf.lite.TFLiteConverter.from_saved_model("model_tf")
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)