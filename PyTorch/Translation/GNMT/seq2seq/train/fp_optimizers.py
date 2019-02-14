import logging
import math

import torch
from torch.nn.utils import clip_grad_norm_


class Fp16Optimizer:
    """
    Mixed precision optimizer with dynamic loss scaling and backoff.
    https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#scalefactor
    """
    @staticmethod
    def set_grads(params, params_with_grad):
        """
        Copies gradients from param_with_grad to params

        :param params: dst parameters
        :param params_with_grad: src parameters
        """
        for param, param_w_grad in zip(params, params_with_grad):
            if param.grad is None:
                param.grad = torch.nn.Parameter(torch.empty_like(param))
            param.grad.data.copy_(param_w_grad.grad.data)

    @staticmethod
    def set_weights(params, new_params):
        """
        Copies parameters from new_params to params

        :param params: dst parameters
        :param new_params: src parameters
        """
        for param, new_param in zip(params, new_params):
            param.data.copy_(new_param.data)

    def __init__(self, fp16_model, grad_clip=float('inf'), loss_scale=8192,
                 dls_downscale=2, dls_upscale=2, dls_upscale_interval=128):
        """
        Constructor for the Fp16Optimizer.

        :param fp16_model: model (previously casted to half)
        :param grad_clip: coefficient for gradient clipping, max L2 norm of the
            gradients
        :param loss_scale: initial loss scale
        :param dls_downscale: loss downscale factor, loss scale is divided by
            this factor when NaN/INF occurs in the gradients
        :param dls_upscale: loss upscale factor, loss scale is multiplied by
            this factor if previous dls_upscale_interval batches finished
            successfully
        :param dls_upscale_interval: interval for loss scale upscaling
        """
        logging.info('Initializing fp16 optimizer')
        self.initialize_model(fp16_model)

        self.since_last_invalid = 0
        self.loss_scale = loss_scale
        self.dls_downscale = dls_downscale
        self.dls_upscale = dls_upscale
        self.dls_upscale_interval = dls_upscale_interval
        self.grad_clip = grad_clip

    def initialize_model(self, model):
        """
        Initializes internal state and build fp32 master copy of weights.

        :param model: fp16 model
        """
        logging.info('Initializing fp32 clone weights')
        self.fp16_model = model
        self.fp16_model.zero_grad()
        self.fp32_params = [param.to(torch.float32).detach()
                            for param in model.parameters()]

        for param in self.fp32_params:
            param.requires_grad = True

    def step(self, loss, optimizer, scheduler, update=True):
        """
        Performs one step of the optimizer.
        Applies loss scaling, computes gradients in fp16, converts gradients to
        fp32, inverts scaling and applies optional gradient norm clipping.
        If gradients are finite, it applies update to fp32 master weights and
        copies updated parameters to fp16 model for the next iteration. If
        gradients are not finite, it skips the batch and adjusts scaling factor
        for the next iteration.

        :param loss: value of loss function
        :param optimizer: optimizer
        :param update: if True executes weight update
        """
        loss *= self.loss_scale
        loss.backward()

        if update:
            self.set_grads(self.fp32_params, self.fp16_model.parameters())
            if self.loss_scale != 1.0:
                for param in self.fp32_params:
                    param.grad.data /= self.loss_scale

            norm = clip_grad_norm_(self.fp32_params, self.grad_clip)

            if math.isfinite(norm):
                scheduler.step()
                optimizer.step()
                self.set_weights(self.fp16_model.parameters(),
                                 self.fp32_params)
                self.since_last_invalid += 1
            else:
                self.loss_scale /= self.dls_downscale
                self.since_last_invalid = 0
                logging.info(f'Gradient norm: {norm}')
                logging.info(f'Skipped batch, new scale: {self.loss_scale}')

            if self.since_last_invalid >= self.dls_upscale_interval:
                self.loss_scale *= self.dls_upscale
                self.loss_scale = min(self.loss_scale, 8192.0)
                logging.info(f'Upscaling, new scale: {self.loss_scale}')
                self.since_last_invalid = 0

            self.fp16_model.zero_grad()


class Fp32Optimizer:
    """
    Standard optimizer, computes backward and applies weight update.
    """
    def __init__(self, model, grad_clip=None):
        """
        Constructor for the Fp32Optimizer

        :param model: model
        :param grad_clip: coefficient for gradient clipping, max L2 norm of the
            gradients
        """
        logging.info('Initializing fp32 optimizer')
        self.initialize_model(model)
        self.grad_clip = grad_clip

    def initialize_model(self, model):
        """
        Initializes state of the model.

        :param model: model
        """
        self.model = model
        self.model.zero_grad()

    def step(self, loss, optimizer, scheduler, update=True):
        """
        Performs one step of the optimizer.

        :param loss: value of loss function
        :param optimizer: optimizer
        :param update: if True executes weight update
        """
        loss.backward()
        if update:
            if self.grad_clip != float('inf'):
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
            scheduler.step()
            optimizer.step()
            self.model.zero_grad()
