#!/bin/sh

TF_DIR='/Users/lichen/Documents/work/others/TensorFlow/tensorflow/'
FZ_TOOL=$TF_DIR"tensorflow/python/tools/freeze_graph.py"
QT_TOOL=$TF_DIR'tensorflow/contrib/quantization/tools/quantize_graph.py'

# WORK_DIR=$1
# OUTPUT_DIR=$2
# OUTPUT_NAME=$3

WORK_DIR='./model'
OUTPUT_DIR='./model_quantized'
OUTPUT_NAME='outputs,initial_state,final_state'

# freeze graph: 把模型与参数合在一起
python $FZ_TOOL \
--input_graph=$WORK_DIR"/graph_def" \
--input_checkpoint=$WORK_DIR"/checkpoint.ckpt" \
--output_graph=$OUTPUT_DIR"/graph_with_var.pb" \
--output_node_names="$OUTPUT_NAME" \
--input_binary

# quantize graph: 量子化参数，压缩模型
python $QT_TOOL \
--input=$OUTPUT_DIR"/graph_with_var.pb" \
--output_node_names="$OUTPUT_NAME" --output=$OUTPUT_DIR"/model.quantized.pb" \
--mode=weights