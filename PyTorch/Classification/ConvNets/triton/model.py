import torch

def update_argparser(parser):
    parser.add_argument(
        "--config", default="resnet50", type=str, required=True, help="Network to deploy")
    parser.add_argument(
        "--checkpoint", default=None, type=str, help="The checkpoint of the model. ")
    parser.add_argument("--classes", type=int, default=1000, help="Number of classes")
    parser.add_argument("--precision", type=str, default="fp32", 
                        choices=["fp32", "fp16"], help="Inference precision")

def get_model(**model_args):
    from image_classification import models

    model = models.resnet50()

    if "checkpoint" in model_args:
        print(f"loading checkpoint {model_args['checkpoint']}")
        state_dict = torch.load(model_args["checkpoint"], map_location="cpu")
        model.load_state_dict({k.replace("module.", ""): v 
                              for k, v in state_dict.items()})
    if model_args["precision"] == "fp16":
        model = model.half()

    model = model.cuda()
    model.eval()
    tensor_names = {"inputs": ["INPUT__0"],
                    "outputs": ["OUTPUT__0"]}

    return model, tensor_names

    
