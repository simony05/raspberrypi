from pathlib import Path
import onnx
from scc4onnx import order_conversion
from onnx_tf.backend import prepare