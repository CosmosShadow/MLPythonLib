#!/bin/sh

TF_DIR='/Users/lichen/Documents/work/SourceOpenOthers/TensorFlow/tensorflow/'
FZ_TOOL=$TF_DIR"tensorflow/python/tools/freeze_graph.py"
QT_TOOL=$TF_DIR'tensorflow/contrib/quantization/tools/quantize_graph.py'
MY_DIR="./models/"

# freeze graph: 把模型与参数合在一起

python $FZ_TOOL \
--input_graph=$MY_DIR"mnist_graph_def" \
--input_checkpoint=$MY_DIR"mnist.ckpt" \
--output_graph=$MY_DIR"mnist_graph_with_var.pb" \
--output_node_names="eval_prediction" \
--input_binary

python $QT_TOOL \
--input=$MY_DIR"mnist_graph_with_var.pb" \
--output_node_names="eval_prediction" --output=$MY_DIR"mnist.quantized.pb" \
--mode=weights
# quantize graph: 量子化参数，压缩模型