from nnunet.nn_unet import NNUnet


def get_model(*, checkpoint_dir: str, precision: str, data_dir: str):
    model = NNUnet.load_from_checkpoint(checkpoint_dir, data_dir=data_dir, triton=True, strict=False)
    model = model.cuda()
    if "fp16" in precision:
        model = model.half()
    model.eval()
    tensor_names = {"inputs": ["INPUT__0"], "outputs": ["OUTPUT__0"]}
    return model, tensor_names
