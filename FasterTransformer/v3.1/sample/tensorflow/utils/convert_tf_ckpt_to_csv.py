import numpy as np
import argparse
import tensorflow as tf

"""
This file converts the model of TensorFlow checkpoint to numpy array,
and store the numpy into <.csv> file. The file name is the name of 
the weights by replacing the '/' by '.'.

For example, the weights of variable 'model/transformer/ffn/kernel:0'
would be stored in file 'model.transformer.ffn.kernel.csv'. And the 
files would be put into the saved_dir.

This converter would skip the variables about training. For example, 
the weights come from Adam optimizers. 
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-saved_dir', '-o', type=str, help='file name of output file', required=True)
    parser.add_argument('-in_file', '-i', type=str, help='file name of input checkpoint file', required=True)
    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))
    print("========================================")

    ckpt_name = args.in_file
    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(ckpt_name + ".meta")
        saver.restore(sess, (ckpt_name))
        all_variables = tf.trainable_variables()
        ckpt = {}
        
        all_val = sess.run(all_variables)
        for var, val in zip(all_variables, all_val):
            if var.name.find("Adam") == -1:
                # ckpt[var.name] = val
                saved_path = args.saved_dir + "/" + var.name[:-2].replace("/", ".") + ".csv"
                np.savetxt(saved_path, np.squeeze(val).astype(np.float32), delimiter=",", fmt="%.10f")

