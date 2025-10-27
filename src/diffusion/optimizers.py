"""
Optimizers for GameNGen
Paper uses Adafactor for Tier 3 (Section 4.2)
"""

from typing import Iterable, Optional, Tuple

import numpy as np
import torch


class Adafactor(torch.optim.Optimizer):
    """
    Adafactor optimizer
    Paper Section 4.2: "with the Adafactor optimizer without weight decay"

    Implementation based on Shazeer & Stern (2018)
    "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost"
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        eps: Tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = True,
    ):
        """
        Args:
            params: Iterable of parameters
            lr: Learning rate (if relative_step=False)
            eps: Regularization constants (eps0, eps1)
            clip_threshold: Threshold for clipping update
            decay_rate: Coefficient for computing running avg of squared gradient
            beta1: Coefficient for running avg of gradient (momentum)
            weight_decay: Weight decay coefficient
            scale_parameter: Whether to scale parameters
            relative_step: Whether to use relative step sizes
            warmup_init: Whether to use warmup initialization
        """
        if relative_step and warmup_init:
            # Paper doesn't use warmup, but option available
            lr = None

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super().__init__(params, defaults)

    def _get_lr(self, param_group, param_scale):
        """Compute learning rate"""
        if param_group["lr"] is None:
            min_step = (
                1e-6 * param_group["step"] if param_group["warmup_init"] else 1e-2
            )
            rel_step_sz = min(min_step, 1.0 / np.sqrt(param_group["step"]))
            param_group["lr"] = rel_step_sz * param_scale
        return param_group["lr"]

    def _get_options(self, param_group, param_shape):
        """Get factorization options"""
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"]
        return factored, use_first_moment

    def _rms(self, tensor):
        """Root mean square"""
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        """Approximation of exponential moving average of square of gradient"""
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt_()
            .unsqueeze(1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(0).rsqrt()
        v = r_factor * c_factor
        return v

    def step(self, closure=None):
        """
        Perform a single optimization step

        Args:
            closure: A closure that reevaluates model and returns loss
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients")

                state = self.state[p]
                grad_shape = grad.shape

                param_group = group
                factored, use_first_moment = self._get_options(param_group, grad_shape)

                # State initialization
                if len(state) == 0:
                    state["step"] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)

                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[0])
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[1:]).flatten()
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0

                state["step"] += 1
                lr = self._get_lr(param_group, self._rms(p.data))
                param_group["lr"] = lr

                # Momentum
                if use_first_moment:
                    state["exp_avg"].mul_(param_group["beta1"]).add_(
                        grad, alpha=1 - param_group["beta1"]
                    )

                # Second moment estimation
                eps_0, eps_1 = param_group["eps"]
                decay_rate = 1 - (state["step"] + 1) ** param_group["decay_rate"]

                if factored:
                    update = grad**2 + eps_0
                    state["exp_avg_sq_row"].mul_(decay_rate).add_(
                        update.mean(dim=list(range(1, len(grad_shape)))),
                        alpha=1 - decay_rate,
                    )
                    state["exp_avg_sq_col"].mul_(decay_rate).add_(
                        update.mean(dim=0).flatten(), alpha=1 - decay_rate
                    )
                    update = self._approx_sq_grad(
                        state["exp_avg_sq_row"], state["exp_avg_sq_col"]
                    )
                    update.mul_(grad)
                else:
                    state["exp_avg_sq"].mul_(decay_rate).add_(
                        grad**2 + eps_0, alpha=1 - decay_rate
                    )
                    update = grad / (state["exp_avg_sq"].sqrt() + eps_1)

                # Clip
                rms = self._rms(update)
                state["RMS"] = rms

                if param_group["clip_threshold"] > 0:
                    clip_val = param_group["clip_threshold"] / max(1.0, rms)
                    update.clamp_(-clip_val, clip_val)

                # Update
                if use_first_moment:
                    update = state["exp_avg"]

                if param_group["weight_decay"] > 0:
                    p.data.mul_(1 - param_group["weight_decay"] * lr)

                p.data.add_(update, alpha=-lr)

        return loss


# Simple wrapper for easier usage
def create_optimizer(optimizer_name: str, parameters, config: dict):
    """
    Create optimizer based on config

    Args:
        optimizer_name: 'AdamW' or 'Adafactor'
        parameters: Model parameters
        config: Configuration dict

    Returns:
        Optimizer instance
    """
    if optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.0),
            betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
            eps=config.get("adam_epsilon", 1e-8),
        )

    elif optimizer_name.lower() == "adafactor":
        # Paper's setting: "without weight decay"
        return Adafactor(
            parameters,
            lr=config["learning_rate"],
            weight_decay=0.0,  # Paper doesn't use weight decay
            scale_parameter=False,
            relative_step=False,
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


if __name__ == "__main__":
    # Test Adafactor
    print("Testing Adafactor optimizer...")

    # Create dummy model
    model = torch.nn.Linear(100, 10)

    # Create optimizer
    optimizer = Adafactor(model.parameters(), lr=2e-5)

    print(f"Optimizer created: {optimizer}")

    # Test step
    x = torch.randn(32, 100)
    y = torch.randn(32, 10)

    for i in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()

        print(f"Step {i}: loss = {loss.item():.4f}")

    print("\n✓ Adafactor test passed!")
