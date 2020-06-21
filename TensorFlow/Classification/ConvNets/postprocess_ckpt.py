import tensorflow as tf
import pdb
import numpy as np
import argparse
import os
import shutil

def main(args):
    with tf.Session() as sess:
        ckpt = args.ckpt
        new_ckpt=args.out
        output_dir = "./new_ckpt_dir"
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        # Create an output directory
        os.mkdir(output_dir)

        new_ckpt_path = os.path.join(output_dir, new_ckpt)
        with open(os.path.join(output_dir, "checkpoint"), 'w') as file:
            file.write("model_checkpoint_path: "+ "\"" + new_ckpt + "\"")
        file.close()
        # Load all the variables
        all_vars = tf.train.list_variables(ckpt)
        ckpt_reader = tf.train.load_checkpoint(ckpt)
        # Capture the dense layer weights and reshape them to a 4D tensor which would be 
        # the weights of a 1x1 convolution layer. This code replaces the dense (FC) layer
        # to a 1x1 conv layer. 
        dense_layer = 'resnet50_v1.5/output/dense/kernel'
        dense_layer_value=0.
        new_var_list=[]
        for var in all_vars:
            curr_var = tf.train.load_variable(ckpt, var[0])
            if var[0]==dense_layer:
                dense_layer_value = curr_var
            else:
                new_var_list.append(tf.Variable(curr_var, name=var[0]))

        new_var_value = np.reshape(dense_layer_value, [1, 1, 2048, 1001])
        new_var = tf.Variable(new_var_value, name=dense_layer)
        new_var_list.append(new_var)
        
        sess.run(tf.global_variables_initializer())
        tf.train.Saver(var_list=new_var_list).save(sess, new_ckpt_path, write_meta_graph=False, write_state=False)
        print ("Rewriting checkpoints completed")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out', type=str, default='./new.ckpt')
    args = parser.parse_args()
    main(args)