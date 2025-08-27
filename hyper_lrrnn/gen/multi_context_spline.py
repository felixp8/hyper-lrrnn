import torch
import torch.nn as nn
from torch.distributions import Transform
from typing import Any

from zuko.flows.autoregressive import MaskedAutoregressiveTransform
from zuko.distributions import DiagNormal
from zuko.lazy import Flow, LazyTransform, UnconditionalDistribution
from zuko.transforms import ComposedTransform, MonotonicRQSTransform


class LazyMultiContextComposedTransform(LazyTransform):
    def __init__(self, *transforms: LazyTransform, n_contexts=2):
        super().__init__()

        self.transforms = nn.ModuleList(transforms)
        self.n_contexts = n_contexts

    def __repr__(self) -> str:
        return repr(self.transforms).replace("ModuleList", "LazyComposedTransform", 1)

    def forward(self, c: Any = None) -> Transform:
        r"""
        Arguments:
            c: A context :math:`c`.

        Returns:
            A transformation :math:`y = f_n \circ \dots \circ f_0(x | c)`.
        """
        base = torch.arange(c.shape[-1], device=c.device) // (c.shape[-1] / self.n_contexts)
        mask_list = [(base == i).to(torch.float) for i in range(self.n_contexts)]
        c_list = [c * mask[None, :] for mask in mask_list]
        return ComposedTransform(*(t(c_list[i % len(c_list)]) for i, t in enumerate(self.transforms)))


class MultiContextMAF(Flow):
    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        randperm: bool = False,
        **kwargs,
    ):
        orders = [
            torch.arange(features),
            torch.flipud(torch.arange(features)),
        ]

        transforms = [
            MaskedAutoregressiveTransform(
                features=features,
                context=context,
                order=torch.randperm(features) if randperm else orders[i % 2],
                **kwargs,
            )
            for i in range(transforms)
        ]
        transforms = LazyMultiContextComposedTransform(*transforms, n_contexts=2)

        base = UnconditionalDistribution(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        super().__init__(transforms, base)


class MultiContextNSF(MultiContextMAF):
    def __init__(
        self,
        features: int,
        context: int = 0,
        bins: int = 8,
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=MonotonicRQSTransform,
            shapes=[(bins,), (bins,), (bins - 1,)],
            **kwargs,
        )