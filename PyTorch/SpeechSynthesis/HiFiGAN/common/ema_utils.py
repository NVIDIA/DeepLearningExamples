import amp_C
import torch


def apply_ema_decay(model, ema_model, decay):
    if not decay:
        return
    st = model.state_dict()
    add_module = hasattr(model, 'module') and not hasattr(ema_model, 'module')
    for k, v in ema_model.state_dict().items():
        if add_module and not k.startswith('module.'):
            k = 'module.' + k
        v.copy_(decay * v + (1 - decay) * st[k])


def init_multi_tensor_ema(model, ema_model):
    model_weights = list(model.state_dict().values())
    ema_model_weights = list(ema_model.state_dict().values())
    ema_overflow_buf = torch.cuda.IntTensor([0])
    return model_weights, ema_model_weights, ema_overflow_buf


def apply_multi_tensor_ema(decay, model_weights, ema_weights, overflow_buf):
    amp_C.multi_tensor_axpby(
        65536, overflow_buf, [ema_weights, model_weights, ema_weights],
        decay, 1-decay, -1)
