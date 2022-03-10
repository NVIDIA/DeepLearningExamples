import torch
from copy import deepcopy
from itertools import chain
from collections import defaultdict, abc as container_abcs

from apex.multi_tensor_apply import multi_tensor_applier

import fused_lamb_CUDA


class FusedLAMBAMP(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, step=0, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01,
                 amsgrad=False, adam_w_mode=True,
                 grad_averaging=True, set_grad_none=True,
                 max_grad_norm=1.0, use_nvlamb=False):
        if amsgrad:
            raise RuntimeError('FusedLAMB does not support the AMSGrad variant.')

        # The learning rate (lr) and optimizer step (step) should be located on device
        # in order to faciliated device sync free execution
        defaults = dict(lr=torch.tensor(lr, dtype=torch.float32, device=torch.cuda.current_device()),
                        step=torch.tensor([step], dtype=torch.int, device=torch.cuda.current_device()),
                        bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        grad_averaging=grad_averaging,
                        max_grad_norm=max_grad_norm)
        super(FusedLAMBAMP, self).__init__(params, defaults)
        if multi_tensor_applier.available:
            import amp_C
            self.multi_tensor_l2norm=amp_C.multi_tensor_l2norm
            # Skip buffer
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device=self.param_groups[0]["params"][0].device)
            self.multi_tensor_lamb = amp_C.multi_tensor_lamb
        else:
            raise RuntimeError('apex.optimizers.FusedLAMB requires cuda extensions')
        
        self._step_supports_amp_scaling = True
        self.param_groups_fp32 = []
        self.adam_w_mode = 1 if adam_w_mode else 0
        self.set_grad_none = set_grad_none
        self.use_nvlamb = use_nvlamb

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain.from_iterable((g['params'] for g in saved_groups)),
                      chain.from_iterable((g['params'] for g in groups)))}

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if param.is_floating_point():
                    value = value.to(value.dtype)
                value = value.to(value.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    def setup_fp32_params(self):
        for i, pg in enumerate(self.param_groups):
            param_list = pg['params']
            self.param_groups_fp32.append({
                'params': [
                    p.clone().detach().float()
                    if p.dtype == torch.half else None
                    for p in param_list
                ],
            })

    def zero_grad(self, set_to_none=False):
        for group in self.param_groups:
            for p in group['params']:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None, grad_scaler=None):
        loss = None
        if closure is not None:
            loss = closure()

        # create separate grad lists for fp32 and fp16 params
        g_all_32, g_all_16 = [], []
        for gid, (group, fp32_group) in enumerate(zip(self.param_groups, self.param_groups_fp32)):
            for pid, (p, fp32_p) in enumerate(zip(group['params'], fp32_group['params'])):
                if p.grad is None:
                    continue
                assert p.dtype in (torch.float16, torch.float32)
                if p.dtype == torch.float32:
                    g_all_32.append(p.grad)
                else:  # p.dtype == torch.float16:
                    g_all_16.append(p.grad)
        device = self.param_groups[0]["params"][0].device
        found_inf = (
            grad_scaler._check_inf_per_device(self)[device]
            if grad_scaler is not None else torch.zeros((1,), device=device)
        )
        self._dummy_overflow_buf.copy_(found_inf)
        scale, inv_scale = None, None
        if grad_scaler:
            scale = grad_scaler._get_scale_async()
            inv_scale = scale.double().reciprocal().float()
        else:
            scale = torch.ones((1,), device=device)
            inv_scale = torch.ones((1,), device=device)
        # g_norm_32, g_norm_16 = torch.zeros(1, device=device), torch.zeros(1, device=device)
        g_norm_32, g_norm_16 = None, None
        # compute grad norm for two lists
        # NOTE(mkozuki): g_all_16, g_all_32, and global_grad_norm are norms of scaled gradients.
        # So, multiply `max_grad_norm` by scale.
        max_grad_norm = self.defaults['max_grad_norm'] * scale
        if len(g_all_32) > 0:
            g_norm_32 = multi_tensor_applier(
                fused_lamb_CUDA.multi_tensor_l2norm,
                self._dummy_overflow_buf,
                [g_all_32],
                False,
            )[0]
        else:
            g_norm_32 = torch.zeros((1,), device=device)
        if len(g_all_16) > 0:
            g_norm_16 = multi_tensor_applier(
                fused_lamb_CUDA.multi_tensor_l2norm,
                self._dummy_overflow_buf,
                [g_all_16],
                False,
            )[0]
        else:
            g_norm_16 = torch.zeros((1,), device=device)

        # blend two grad norms to get global grad norm
        global_grad_norm = multi_tensor_applier(
            fused_lamb_CUDA.multi_tensor_l2norm,
            self._dummy_overflow_buf,
            [[g_norm_32, g_norm_16]],
            False,
        )[0]

        # Run LAMB optimization math
        for gid, (group, fp32_group) in enumerate(zip(self.param_groups, self.param_groups_fp32)):
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']
            grad_averaging = 1 if group['grad_averaging'] else 0

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += (self._dummy_overflow_buf != 1).int()
            else:
                group['step'] = (self._dummy_overflow_buf != 1).int()

            # create lists for multi-tensor apply
            g_16, p_16, m_16, v_16, dst_param_fp16 = [], [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []

            for p, p_fp32 in zip(group['params'], fp32_group['params']):
                if p.grad is None:
                    continue
                assert not p.grad.is_sparse

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    dtype = torch.float if p.dtype == torch.half else p.dtype
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=dtype)
                    # Exponential moving average of gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=dtype)

                if p.dtype == torch.float16:
                    g_16.append(p.grad.data)
                    p_16.append(p_fp32.data)
                    m_16.append(state['exp_avg'])
                    v_16.append(state['exp_avg_sq'])
                    dst_param_fp16.append(p.data)
                elif p.dtype == torch.float32:
                    assert p_fp32 is None
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(state['exp_avg'])
                    v_32.append(state['exp_avg_sq'])
                else:
                    raise RuntimeError('FusedLAMB only support fp16 and fp32.')

            if g_16:
                multi_tensor_applier(
                    fused_lamb_CUDA.multi_tensor_lamb,
                    self._dummy_overflow_buf,
                    [g_16, p_16, m_16, v_16, dst_param_fp16],
                    group['lr'], beta1, beta2, group['eps'],
                    group['step'], bias_correction, group['weight_decay'],
                    grad_averaging, self.adam_w_mode,
                    global_grad_norm, max_grad_norm, self.use_nvlamb,
                    found_inf, inv_scale)
            if g_32:
                multi_tensor_applier(
                    fused_lamb_CUDA.multi_tensor_lamb,
                    self._dummy_overflow_buf,
                    [g_32, p_32, m_32, v_32],
                    group['lr'], beta1, beta2, group['eps'],
                    group['step'], bias_correction, group['weight_decay'],
                    grad_averaging, self.adam_w_mode,
                    global_grad_norm, max_grad_norm, self.use_nvlamb,
                    found_inf, inv_scale)

        return loss

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.
        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.
        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
            specific optimization options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if isinstance(default, torch.Tensor) : 
                param_group.setdefault(name, deepcopy(default))
            else :
                param_group.setdefault(name, default)

        params = param_group['params']
        if len(params) != len(set(params)):
            warnings.warn("optimizer contains a parameter group with duplicate parameters; "
                          "in future, this will cause an error; "
                          "see github.com/pytorch/pytorch/issues/40967 for more information", stacklevel=3)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)
