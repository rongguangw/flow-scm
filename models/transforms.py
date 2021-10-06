import pyro
import torch
from torch.distributions import transforms
from pyro.distributions.torch_transform import TransformModule
from pyro.nn import AutoRegressiveNN, ConditionalAutoRegressiveNN
from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.transforms import SplineAutoregressive, ConditionalSplineAutoregressive


class ConditionalAffineTransform(ConditionalTransformModule):
    def __init__(self, context_nn, event_dim=0, **kwargs):
        super().__init__(**kwargs)

        self.event_dim = event_dim
        self.context_nn = context_nn

    def condition(self, context):
        loc, log_scale = self.context_nn(context)
        scale = torch.exp(log_scale)

        ac = transforms.AffineTransform(loc, scale, event_dim=self.event_dim)
        return ac


def spline_autoregressive(input_dim, hidden_dims=None, count_bins=8, bound=3.0, order='linear'):
    r"""
    A helper function to create an
    :class:`~pyro.distributions.transforms.SplineAutoregressive` object that takes
    care of constructing an autoregressive network with the correct input/output
    dimensions.
    :param input_dim: Dimension of input variable
    :type input_dim: int
    :param hidden_dims: The desired hidden dimensions of the autoregressive network.
        Defaults to using [3*input_dim + 1]
    :type hidden_dims: list[int]
    :param count_bins: The number of segments comprising the spline.
    :type count_bins: int
    :param bound: The quantity :math:`K` determining the bounding box,
        :math:`[-K,K]\times[-K,K]`, of the spline.
    :type bound: float
    :param order: One of ['linear', 'quadratic'] specifying the order of the spline.
    :type order: string
    """

    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]

    if order == 'linear':
        param_dims = [count_bins, count_bins, count_bins - 1, count_bins]
    elif order == 'quadratic':
        param_dims = [count_bins, count_bins, count_bins - 1]

    arn = AutoRegressiveNN(input_dim, hidden_dims, param_dims=param_dims)
    return SplineAutoregressive(input_dim, arn, count_bins=count_bins, bound=bound, order=order)


def conditional_spline_autoregressive(input_dim, context_dim, hidden_dims=None, count_bins=8, bound=3.0, order='linear'):
    r"""
    A helper function to create a
    :class:`~pyro.distributions.transforms.ConditionalSplineAutoregressive` object
    that takes care of constructing an autoregressive network with the correct
    input/output dimensions.
    :param input_dim: Dimension of input variable
    :type input_dim: int
    :param context_dim: Dimension of context variable
    :type context_dim: int
    :param hidden_dims: The desired hidden dimensions of the autoregressive network.
        Defaults to using [input_dim * 10, input_dim * 10]
    :type hidden_dims: list[int]
    :param count_bins: The number of segments comprising the spline.
    :type count_bins: int
    :param bound: The quantity :math:`K` determining the bounding box,
        :math:`[-K,K]\times[-K,K]`, of the spline.
    :type bound: float
    :param order: One of ['linear', 'quadratic'] specifying the order of the spline.
    :type order: string
    """

    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]

    if order == 'linear':
        param_dims = [count_bins, count_bins, count_bins - 1, count_bins]
    elif order == 'quadratic':
        param_dims = [count_bins, count_bins, count_bins - 1]

    arn = ConditionalAutoRegressiveNN(input_dim, context_dim, hidden_dims, param_dims=param_dims)
    return ConditionalSplineAutoregressive(input_dim, arn, count_bins=count_bins, bound=bound, order=order)
