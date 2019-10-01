"""loss function utilities."""
import torch
import torch.nn.functional as F


def compute_advantages(discount, gae_lambda, max_len, baselines, rewards):
    """
    Calculate advantages.

    Advantages are a discounted cumulative sum.

    The discount cumulative sum can be represented as an IIR
    filter ob the reversed input vectors, i.e.
       y[t] - discount*y[t+1] = x[t]
           or
       rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]

    Given the time-domain IIR filter step response, we can
    calculate the filter response to our signal by convolving the
    signal with the filter response function. The time-domain IIR
    step response is calculated below as discount_filter:
        discount_filter =
            [1, discount, discount^2, ..., discount^N-1]
            where the epsiode length is N.
    """
    filter = torch.full((1, 1, 1, max_len - 1), discount * gae_lambda)
    filter = torch.cumprod(F.pad(filter, (1, 0), value=1), dim=-1)

    deltas = (rewards + discount * F.pad(baselines, (0, 1))[:, 1:] - baselines)
    deltas = F.pad(deltas, (0, max_len - 1)).unsqueeze(0).unsqueeze(0)

    advantages = F.conv2d(deltas, filter, stride=1).squeeze()
    return advantages


def pad_to_last(nums, total_length, axis=-1, val=0):
    """
    Pad val to last in nums in given axis.

    length of the result in given axis should be total_length.

    Args:
        nums:
        total_length:
        axis:
        val:

    Returns:

    """
    tensor = torch.Tensor(nums)
    axis = (axis + len(tensor.shape)) if axis < 0 else axis

    if len(tensor.shape) <= axis:
        raise IndexError(f'axis {axis} is out of range {tensor.shape}')

    padding_config = [0, 0] * len(tensor.shape)
    padding_idx = abs(axis - len(tensor.shape)) * 2 - 1
    padding_config[padding_idx] = max(total_length - tensor.shape[axis], val)
    return F.pad(tensor, padding_config)


def filter_valids(tensor, valids):
    """
    Filter out tensor using valids (last index of valid tensors).

    valids contains last indices of each rows.

    Args:
        tensor:
        valids:

    Returns:

    """
    return [tensor[i][:valids[i]] for i in range(len(valids))]
