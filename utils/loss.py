import torch
import math

def gaussian_nll_loss(
    input,
    target,
    var,
    full: bool = False,
    eps: float = 1e-6,
    reduction: str = "mean",
):
    r"""Gaussian negative log likelihood loss.

    See :class:`~torch.nn.GaussianNLLLoss` for details.

    Args:
        input: expectation of the Gaussian distribution.
        target: sample from the Gaussian distribution.
        var: tensor of positive variance(s), one for each of the expectations
            in the input (heteroscedastic), or a single one (homoscedastic).
        full (bool, optional): include the constant term in the loss calculation. Default: ``False``.
        eps (float, optional): value added to var, for stability. Default: 1e-6.
        reduction (string, optional): specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output is the average of all batch member losses,
            ``'sum'``: the output is the sum of all batch member losses.
            Default: ``'mean'``.
    """
    # Check var size
    # If var.size == input.size, the case is heteroscedastic and no further checks are needed.
    # Otherwise:
    if var.size() != input.size():

        # If var is one dimension short of input, but the sizes match otherwise, then this is a homoscedastic case.
        # e.g. input.size = (10, 2, 3), var.size = (10, 2)
        # -> unsqueeze var so that var.shape = (10, 2, 1)
        # this is done so that broadcasting can happen in the loss calculation
        if input.size()[:-1] == var.size():
            var = torch.unsqueeze(var, -1)

        # This checks if the sizes match up to the final dimension, and the final dimension of var is of size 1.
        # This is also a homoscedastic case.
        # e.g. input.size = (10, 2, 3), var.size = (10, 2, 1)
        elif input.size()[:-1] == var.size()[:-1] and var.size(-1) == 1:  # Heteroscedastic case
            pass

        # If none of the above pass, then the size of var is incorrect.
        else:
            raise ValueError("var is of incorrect size")

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    # Calculate the loss
    loss = 0.5 * (torch.log(var) + (input - target)**2 / var)
    if full:
        loss += 0.5 * math.log(2 * math.pi)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss