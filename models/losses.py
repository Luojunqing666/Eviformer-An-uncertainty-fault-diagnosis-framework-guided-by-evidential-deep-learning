"""Eviformer: Evidential deep learning losses for uncertainty quantification."""

import torch
import torch.nn.functional as F


def relu_evidence(y):
    """Compute evidence using ReLU activation."""
    return F.relu(y)


def exp_evidence(y):
    """Compute evidence using clamped exponential activation."""
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    """Compute evidence using Softplus activation."""
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    """
    Compute KL divergence between a Dirichlet distribution and a uniform Dirichlet prior.

    Args:
        alpha: Dirichlet concentration parameters [batch_size, num_classes].
        num_classes: Number of classes.
        device: Computation device.

    Returns:
        KL divergence for each sample [batch_size, 1].
    """
    if device is None:
        device = alpha.device
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device=None):
    """
    Compute the log-likelihood loss for Dirichlet distribution.

    Args:
        y: One-hot encoded labels [batch_size, num_classes].
        alpha: Dirichlet concentration parameters [batch_size, num_classes].
        device: Computation device.

    Returns:
        Log-likelihood loss for each sample [batch_size, 1].
    """
    if device is None:
        device = alpha.device
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    """
    Compute MSE-based evidential loss with KL divergence regularization.

    Args:
        y: One-hot encoded labels [batch_size, num_classes].
        alpha: Dirichlet concentration parameters [batch_size, num_classes].
        epoch_num: Current epoch number.
        num_classes: Number of classes.
        annealing_step: Number of annealing steps for KL weight.
        device: Computation device.

    Returns:
        Total loss for each sample [batch_size, 1].
    """
    if device is None:
        device = alpha.device
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    """
    General evidential deep learning loss.

    Args:
        func: Activation function (torch.log or torch.digamma).
        y: One-hot encoded labels [batch_size, num_classes].
        alpha: Dirichlet concentration parameters [batch_size, num_classes].
        epoch_num: Current epoch number.
        num_classes: Number of classes.
        annealing_step: Number of annealing steps for KL weight.
        device: Computation device.

    Returns:
        Total loss for each sample [batch_size, 1].
    """
    if device is None:
        device = alpha.device
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    """
    Evidential MSE loss: uses MSE-based likelihood with KL regularization.

    Args:
        output: Model output (logits) [batch_size, num_classes].
        target: One-hot encoded labels [batch_size, num_classes].
        epoch_num: Current epoch number.
        num_classes: Number of classes.
        annealing_step: Number of annealing steps for KL weight.
        device: Computation device.

    Returns:
        Scalar loss value.
    """
    if device is None:
        device = output.device
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    """
    Evidential log loss: uses negative log of expected likelihood.

    Args:
        output: Model output (logits) [batch_size, num_classes].
        target: One-hot encoded labels [batch_size, num_classes].
        epoch_num: Current epoch number.
        num_classes: Number of classes.
        annealing_step: Number of annealing steps for KL weight.
        device: Computation device.

    Returns:
        Scalar loss value.
    """
    if device is None:
        device = output.device
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(torch.log, target, alpha, epoch_num, num_classes, annealing_step, device)
    )
    return loss


def edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    """
    Evidential digamma loss: uses expected cross entropy via digamma function.

    Args:
        output: Model output (logits) [batch_size, num_classes].
        target: One-hot encoded labels [batch_size, num_classes].
        epoch_num: Current epoch number.
        num_classes: Number of classes.
        annealing_step: Number of annealing steps for KL weight.
        device: Computation device.

    Returns:
        Scalar loss value.
    """
    if device is None:
        device = output.device
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device)
    )
    return loss
