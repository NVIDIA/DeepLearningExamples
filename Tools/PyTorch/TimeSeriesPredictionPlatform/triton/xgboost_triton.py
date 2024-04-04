import os
import numpy
import dllogger
import tritonclient.http as triton_http
import tritonclient.grpc as triton_grpc
import numpy as np
import xgboost as xgb
import hydra
import subprocess

def generate_config(
        model_name,
        *,
        features=31,
        predict_proba=False,
        batching_window=100,
        max_batch_size=8192,
        storage_type="AUTO"):
    """Return a string with the full Triton config.pbtxt for this model
    """
    model_format = 'xgboost'
    instance_kind = 'gpu'
    if instance_kind == 'gpu':
        instance_kind = 'KIND_GPU'
    predict_proba = 'false'
    return f"""name: "{model_name}"
backend: "fil"
max_batch_size: {max_batch_size}
input [
 {{
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ {features} ]
  }}
]
output [
 {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ {1} ]
  }}
]
instance_group [{{ kind: {instance_kind} }}]
parameters [
  {{
    key: "model_type"
    value: {{ string_value: "{model_format}" }}
  }},
  {{
    key: "output_class"
    value: {{ string_value: "{predict_proba}" }}
  }},
  {{
    key: "format"
    value: {{ string_value: "{model_format}"}}
  }}
]

dynamic_batching {{
  max_queue_delay_microseconds: {batching_window}
}}"""

def format_checkpoint(ckpt, total_features, max_batch_size):
    main_output_path = ckpt
    #make deployment
    checkpoint_path = os.path.join(main_output_path, 'checkpoints')
    #make navigator_workspace
    os.makedirs(os.path.join(main_output_path, 'deployment'), exist_ok=True)
    os.makedirs(os.path.join(main_output_path, 'deployment', 'navigator_workspace'), exist_ok=True)
    os.makedirs(os.path.join(main_output_path, 'deployment', 'navigator_workspace', 'model-store'), exist_ok=True)
    #make model-store
    model_store_path = os.path.join(main_output_path, 'deployment', 'navigator_workspace', 'model-store')
    #iterate over the models
    for ckpt in os.listdir(checkpoint_path):
        #load model
        model = xgb.Booster()
        model.load_model(os.path.join(checkpoint_path, ckpt))
        model_name = ckpt.split(".")[0]
        os.makedirs(os.path.join(model_store_path, model_name), exist_ok=True)
        os.makedirs(os.path.join(model_store_path, model_name, '1'), exist_ok=True)
        model.save_model(os.path.join(model_store_path, model_name, '1', 'xgboost.model'))
        #grab the config
        triton_config = generate_config(model_name=model_name, features=total_features, max_batch_size=max_batch_size)
        #put in model-store
        config_path = os.path.join(os.path.join(model_store_path, model_name), 'config.pbtxt')
        with open(config_path, 'w') as config_file:
            config_file.write(triton_config)



def run_XGBoost_triton(cfg, config):
    ckpt = cfg.checkpoint
    max_batch_size = cfg.batch_size
    tspp_main_dir = os.path.sep + os.path.join(*(os.getcwd().split(os.path.sep)[:-3]))
    #need the extra inference stuff
    train, valid, test = hydra.utils.call(config.dataset)
    del train, valid
    dataloader = test
    original_features = len(test.data.columns)
    total_features =  original_features + len(test.known) - len(test.target) + 1
    format_checkpoint(ckpt, total_features, max_batch_size)
