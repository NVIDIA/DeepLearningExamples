import argparse

import torch
import pytorch_quantization


from image_classification.models import (
    resnet50,
    resnext101_32x4d,
    se_resnext101_32x4d,
    efficientnet_b0,
    efficientnet_b4,
    efficientnet_widese_b0,
    efficientnet_widese_b4,
    efficientnet_quant_b0,
    efficientnet_quant_b4,
)


def available_models():
    models = {
        m.name: m
        for m in [
            resnet50,
            resnext101_32x4d,
            se_resnext101_32x4d,
            efficientnet_b0,
            efficientnet_b4,
            efficientnet_widese_b0,
            efficientnet_widese_b4,
            efficientnet_quant_b0,
            efficientnet_quant_b4,
        ]
    }
    return models


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    model_names = available_models().keys()
    parser.add_argument("--arch", "-a", metavar="ARCH", default="resnet50", choices=model_names,
                        help="model architecture: " + " | ".join(model_names) + " (default: resnet50)")
    parser.add_argument("--device", metavar="DEVICE", default="cuda", choices=['cpu', 'cuda'],
                        help="device on which model is settled: cpu, cuda (default: cuda)")
    parser.add_argument("--image-size", default=None, type=int, help="resolution of image")
    parser.add_argument('--output', type=str, help='Path to converted model')
    parser.add_argument("-b", "--batch-size", default=256, type=int, metavar="N",
                        help="mini-batch size (default: 256) per gpu")

    return parser


def final_name(base_name):
    splitted = base_name.split('.')
    if 'pt' in splitted:
        fin_name = base_name.replace('pt', 'onnx')
    elif 'pth' in splitted:
        fin_name = base_name.replace('pth', 'onnx')
    elif len(splitted) > 1:
        fin_name = '.'.join(splitted[:-1] + ['onnx'])
    else:
        fin_name = base_name + '.onnx'
    return fin_name


def get_dataloader(image_size, bs, num_classes):
    """return dataloader for inference"""
    from image_classification.dataloaders import get_syntetic_loader

    def data_loader():
        loader, _ = get_syntetic_loader(None, image_size, bs, num_classes, False)
        for inp, _ in loader:
            yield inp
            break

    return data_loader()


def prepare_inputs(dataloader, device):
    """load sample inputs to device"""
    inputs = []
    for batch in dataloader:
        if type(batch) is torch.Tensor:
            batch_d = batch.to(device)
            batch_d = (batch_d, )
            inputs.append(batch_d)
        else:
            batch_d = []
            for x in batch:
                assert type(x) is torch.Tensor, "input is not a tensor"
                batch_d.append(x.to(device))
            batch_d = tuple(batch_d)
            inputs.append(batch_d)
    return inputs


def check_quant_weight_correctness(checkpoint_path, model):
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    quantizers_sd_keys = {f'{n[0]}._amax' for n in model.named_modules() if 'quantizer' in n[0]}
    sd_all_keys = quantizers_sd_keys | set(model.state_dict().keys())
    assert set(state_dict.keys()) == sd_all_keys, (f'Passed quantized architecture, but following keys are missing in '
                                                   f'checkpoint: {list(sd_all_keys - set(state_dict.keys()))}')


def main(args, model_args, model_arch):

    quant_arch = args.arch in ['efficientnet-quant-b0', 'efficientnet-quant-b4']
    if quant_arch:
        pytorch_quantization.nn.modules.tensor_quantizer.TensorQuantizer.use_fb_fake_quant = True

    model = model_arch(**model_args.__dict__)

    if quant_arch and model_args.pretrained_from_file is not None:
        check_quant_weight_correctness(model_args.pretrained_from_file, model)

    image_size = args.image_size if args.image_size is not None else model.arch.default_image_size

    train_loader = get_dataloader(image_size, args.batch_size, model_args.num_classes)
    inputs = prepare_inputs(train_loader, args.device)
    final_model_path = args.output if args.output is not None else final_name(model_args.pretrained_from_file)
    model.to(args.device)
    model.eval()

    with torch.no_grad():
        torch.onnx.export(model,
                          inputs[0],
                          final_model_path,
                          verbose=True,
                          opset_version=13,
                          enable_onnx_checker=True,
                          do_constant_folding=True)


if __name__ == '__main__':
    epilog = [
        "Based on the architecture picked by --arch flag, you may use the following options:\n"
    ]
    for model, ep in available_models().items():
        model_help = "\n".join(ep.parser().format_help().split("\n")[2:])
        epilog.append(model_help)
    parser = argparse.ArgumentParser(
        description="PyTorch ImageNet Training",
        epilog="\n".join(epilog),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser = parse_args(parser)
    args, rest = parser.parse_known_args()

    model_arch = available_models()[args.arch]
    model_args, rest = model_arch.parser().parse_known_args(rest)

    assert len(rest) == 0, f"Unknown args passed: {rest}"
    main(args, model_args, model_arch)
