import torch

from bgtorch.nn import DenseNet
from bgtorch.nn.flow.transformer import AffineTransformer
from bgtorch.nn.flow import CouplingFlow

# install from https://github.com/bayesiains/nflows/tree/master/nflows 
from nflows.transforms.splines.rational_quadratic import rational_quadratic_spline, unconstrained_rational_quadratic_spline
from bgtorch.nn.flow.transformer.base import Transformer


def RNVP(shift_nhidden, scale_nhidden,
         shift_activation=torch.nn.ReLU(), scale_activation=torch.nn.Tanh(), 
         shift_weight_scale=1.0, scale_weight_scale=1.0, 
         shift_bias_scale=0.0, scale_bias_scale=0.0, init_downscale=1.0):
    
    shift_transformation = None
    if shift_nhidden is not None:
        shift_transformation=DenseNet(shift_nhidden, activation=shift_activation, 
                                      weight_scale=shift_weight_scale, bias_scale=shift_bias_scale)            
    scale_transformation = None
    if scale_nhidden is not None:
        scale_transformation=DenseNet(scale_nhidden, activation=scale_activation, 
                                      weight_scale=scale_weight_scale, bias_scale=scale_bias_scale)
    transformer = AffineTransformer(shift_transformation=shift_transformation,
                                    scale_transformation=scale_transformation,
                                    init_downscale=init_downscale) 
    return CouplingFlow(transformer)


class ConditionalSplineTransform(Transformer):
    """ Advanced transformer based on rational quadratic splines (see NSF paper)"""
    
    def __init__(self, 
                 n_bins : int,
                 width_net : torch.nn.Module,
                 height_net : torch.nn.Module,
                 slope_net  : torch.nn.Module,
                 is_circular : bool=False,
                 tail : float=1.):
        """
            n_bins: number of spline knot points
            width_net: computes x difference between knot points
                       output must be `dim * n_bins`
            height_net: computes y difference between knot points
                        output must be `dim * n_bins`
            slope_net: computes slope at knot points
                       output must be `dim * (n_bins + 1)`
            is_circular: True if target value is an angle
            tail: defines value after which splines are linearly interpolated
                  should be set to something like `max(data.abs())`
        """
        super().__init__()
        self._n_bins = n_bins
        self._width_net = width_net
        self._height_net = height_net
        self._slope_net = slope_net
        self._is_circular = is_circular
        self._tail = tail
        
    def _compute_params(self, x):
        width = self._width_net(x).view(x.shape[0], -1, self._n_bins)
        height = self._height_net(x).view(x.shape[0], -1, self._n_bins)
        slope = self._slope_net(x).view(x.shape[0], -1, self._n_bins + 1)
        if self._is_circular:
            slope[..., -1] = slope[..., 0]
        return width, height, slope
        
    def _forward(self, x, y, *args, **kwargs):
        width, height, slope = self._compute_params(x)
        if self._is_circular:
            y, dlogp = rational_quadratic_spline(y, width, height, slope)
        else:
            y, dlogp = unconstrained_rational_quadratic_spline(y, width, height, slope, tail_bound=self._tail)
        return y, dlogp.sum(dim=-1, keepdim=True)
    
    def _inverse(self, x, y, *args, **kwargs):
        width, height, slope = self._compute_params(x)
        if self._is_circular:
            y, dlogp = rational_quadratic_spline(y, width, height, slope, inverse=True)
        else:
            y, dlogp = unconstrained_rational_quadratic_spline(y, width, height, slope, inverse=True, tail_bound=self._tail)
        return y, dlogp.sum(dim=-1, keepdim=True)
        


def NSF(width_nhidden, height_nhidden, slope_nhidden, n_bins, 
        tail,
        width_activation=torch.nn.ReLU(), height_activation=torch.nn.ReLU(), slope_activation=torch.nn.ReLU(),
        width_weight_scale=1.0, height_weight_scale=1.0, slope_weight_scale=1.0):
    
    width_net = DenseNet(width_nhidden, activation=width_activation, weight_scale=width_weight_scale)
    height_net = DenseNet(height_nhidden, activation=height_activation, weight_scale=height_weight_scale)
    slope_net = DenseNet(slope_nhidden, activation=slope_activation, weight_scale=slope_weight_scale)
    
    transformer = ConditionalSplineTransform(
        width_net=width_net,
        height_net=height_net,
        slope_net=slope_net,
        n_bins=n_bins,
        tail=tail
    ) 
    
    return CouplingFlow(transformer)

