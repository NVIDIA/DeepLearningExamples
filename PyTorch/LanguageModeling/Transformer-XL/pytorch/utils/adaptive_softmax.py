import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLogSoftmax(nn.Module):
    def __init__(self, in_features, n_classes, cutoffs, keep_order=False):
        super(AdaptiveLogSoftmax, self).__init__()

        cutoffs = list(cutoffs)

        if (cutoffs != sorted(cutoffs)) \
                or (min(cutoffs) <= 0) \
                or (max(cutoffs) >= (n_classes - 1)) \
                or (len(set(cutoffs)) != len(cutoffs)) \
                or any([int(c) != c for c in cutoffs]):

            raise ValueError("cutoffs should be a sequence of unique, positive "
                             "integers sorted in an increasing order, where "
                             "each value is between 1 and n_classes-1")

        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.in_features))
        self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

        self.keep_order = keep_order

    def forward(self, hidden, target, weight, bias, keep_order=False):
        if hidden.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        head_weight = torch.cat(
            [weight[:self.shortlist_size], self.cluster_weight], dim=0)
        head_bias = torch.cat(
            [bias[:self.shortlist_size], self.cluster_bias], dim=0)

        head_logit = F.linear(hidden, head_weight, bias=head_bias)
        head_logprob = F.log_softmax(head_logit, dim=1)

        nll = torch.zeros_like(target, dtype=hidden.dtype,
                               device=hidden.device)

        offset = 0
        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):
            l_idx, h_idx = cutoff_values[i], cutoff_values[i + 1]

            mask_i = (target >= l_idx) & (target < h_idx)
            indices_i = mask_i.nonzero(as_tuple=False).squeeze()

            if indices_i.numel() == 0:
                continue

            target_i = target.index_select(0, indices_i) - l_idx
            head_logprob_i = head_logprob.index_select(0, indices_i)

            if i == 0:
                logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
            else:
                weight_i = weight[l_idx:h_idx]
                bias_i = bias[l_idx:h_idx]

                hidden_i = hidden.index_select(0, indices_i)

                tail_logit_i = F.linear(hidden_i, weight_i, bias=bias_i)
                tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)

                logprob_i = head_logprob_i[:, -i] \
                    + tail_logprob_i.gather(1, target_i[:, None]).squeeze(1)

            if (hasattr(self, 'keep_order') and self.keep_order) or keep_order:
                nll.index_copy_(0, indices_i, -logprob_i)
            else:
                nll[offset:offset+logprob_i.size(0)].copy_(-logprob_i)

            offset += logprob_i.size(0)

        return nll
