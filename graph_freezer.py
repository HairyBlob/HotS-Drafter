# Preparing a TF model for usage in Android
# By Omid Alemi - Jan 2017
# Works with TF <r1.0

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import os

# Freeze the graph
def freezing_graph(model):
    dirname = os.path.dirname(__file__)
    checkpoint_path =""
    input_graph_path =""
    input_saver_def_path =""
    output_frozen_graph_name = ""
    output_optimized_graph_name = ""
    output_path = ""
    if model == "estimator":
        checkpoint_path = os.path.join(dirname, "estimator_graph\\estimator_checkpoint-0")
        input_graph_path =  os.path.join(dirname, "estimator_graph\\estimator_graph.pb")
        output_path = os.path.join(dirname, "estimator_graph")
        output_frozen_graph_name = os.path.join(dirname, "estimator_graph\\frozen_estimator.pb")
        output_optimized_graph_name = "optimized_estimator.pb"
        
    elif model == "discriminator":
        checkpoint_path = os.path.join(dirname, "discriminator_graph\\discriminator_checkpoint-0")
        input_graph_path = os.path.join(dirname, "discriminator_graph\\discriminator_graph.pb")
        output_path = os.path.join(dirname, "discriminator_graph")
        output_frozen_graph_name = os.path.join(dirname, "discriminator_graph\\frozen_discriminator.pb")
        output_optimized_graph_name = "optimized_discriminator.pb"
        
    input_binary = False
    output_node_names = "y"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    clear_devices = True
    
    
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_frozen_graph_name, clear_devices, "")
    
    
    
    # Optimize for inference
    
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)
    
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def,
            ["x"], # an array of the input node(s)
            ["y"], # an array of output nodes
            tf.float32.as_datatype_enum)
    
    
    # Save the optimized graph
    
    f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
    f.write(output_graph_def.SerializeToString())
    
    tf.train.write_graph(output_graph_def, output_path, output_optimized_graph_name)