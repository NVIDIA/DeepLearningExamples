from tqdm import tqdm
import torch
import contextlib
import time
import logging

from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from . import logger as log
from .utils import calc_ips
import dllogger

initialize = quant_modules.initialize
deactivate = quant_modules.deactivate

IPS_METADATA = {"unit": "img/s", "format": ":.2f"}
TIME_METADATA = {"unit": "s", "format": ":.5f"}


def select_default_calib_method(calib_method='histogram'):
    """Set up selected calibration method in whole network"""
    quant_desc_input = QuantDescriptor(calib_method=calib_method)
    quant_nn.QuantConv1d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantAdaptiveAvgPool2d.set_default_quant_desc_input(quant_desc_input)


def quantization_setup(calib_method='histogram'):
    """Change network into quantized version "automatically" and selects histogram as default quantization method"""
    select_default_calib_method(calib_method)
    initialize()


def disable_calibration(model):
    """Disables calibration in whole network. Should be run always before running interference."""
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def collect_stats(model, data_loader, logger, num_batches):
    """Feed data to the network and collect statistic"""
    if logger is not None:
        logger.register_metric(
            f"calib.total_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            f"calib.data_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            f"calib.compute_latency",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=TIME_METADATA,
        )
    # Enable calibrators
    data_iter = enumerate(data_loader)
    if logger is not None:
        data_iter = logger.iteration_generator_wrapper(data_iter, mode='calib')

    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    end = time.time()

    if logger is not None:
        logger.start_calibration()

    for i, (image, _) in data_iter:
        bs = image.size(0)
        data_time = time.time() - end

        model(image.cuda())

        it_time = time.time() - end

        if logger is not None:
            logger.log_metric(f"calib.total_ips", calc_ips(bs, it_time))
            logger.log_metric(f"calib.data_time", data_time)
            logger.log_metric(f"calib.compute_latency", it_time - data_time)

        if i >= num_batches:
            time.sleep(5)
            break

        end = time.time()

    if logger is not None:
        logger.end_calibration()

    logging.disable(logging.WARNING)
    disable_calibration(model)


def compute_amax(model, **kwargs):
    """Loads statistics of data and calculates quantization parameters in whole network"""
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer) and module._calibrator is not None:
            if isinstance(module._calibrator, calib.MaxCalibrator):
                module.load_calib_amax()
            else:
                module.load_calib_amax(**kwargs)
    model.cuda()


def calibrate(model, train_loader, logger, calib_iter=1, percentile=99.99):
    """Calibrates whole network i.e. gathers data for quantization and calculates quantization parameters"""
    model.eval()

    with torch.no_grad():
        collect_stats(model, train_loader, logger, num_batches=calib_iter)
        compute_amax(model, method="percentile", percentile=percentile)

    logging.disable(logging.NOTSET)


@contextlib.contextmanager
def switch_on_quantization(do_quantization=True):
    """Context manager for quantization activation"""
    if do_quantization:
        initialize()
    try:
        yield
    finally:
        if do_quantization:
            deactivate()
