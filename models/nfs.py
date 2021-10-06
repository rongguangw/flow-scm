import pyro
import torch
import numpy as np
import torch.nn as nn

from torch.distributions import Independent
from pyro.infer.reparam.transform import TransformReparam
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.nn import PyroModule, pyro_method, DenseNN, AutoRegressiveNN
from pyro.distributions import Normal, Bernoulli, Categorical, TransformedDistribution
from pyro.distributions.transforms import Spline, AffineTransform, ExpTransform, ComposeTransform, spline_coupling, spline_autoregressive
from pyro.distributions.transforms import conditional_spline
from pyro.distributions.conditional import ConditionalTransformedDistribution
from .transforms import ConditionalAffineTransform, conditional_spline_autoregressive


class ConditionalSCM(PyroModule):
    def __init__(
        self,
        age_dim=1,
        sex_dim=1,
        scanner_dim=6,
        context_dim=3,
        roi_dim=145,
        flow_dict=None,
        flow_type='affine',
        spline_bins=8,
        spline_order='linear',
        spline_hidden_dims=None,
        normalize=False
        ):
        super(ConditionalSCM, self).__init__()
        self.sex_dim = sex_dim
        self.age_dim = age_dim
        self.roi_dim = roi_dim
        self.scanner_dim = scanner_dim
        self.context_dim = context_dim
        self.flow_dict = flow_dict
        self.flow_type = flow_type
        self.spline_bins = spline_bins
        self.spline_order = spline_order
        self.spline_hidden_dims = spline_hidden_dims
        self.normalize = normalize

        # sex prior
        self.sex_logits = torch.nn.Parameter(self.flow_dict['sex_logits'].cuda())
        # age priors
        self.age_base_loc = torch.zeros([self.age_dim, ], device='cuda', requires_grad=False)
        self.age_base_scale = torch.ones([self.age_dim, ], device='cuda', requires_grad=False)
        self.register_buffer('age_flow_lognorm_loc', torch.zeros([], requires_grad=False))
        self.register_buffer('age_flow_lognorm_scale', torch.ones([], requires_grad=False))
        # scanner prior
        self.scanner_logits = torch.nn.Parameter(self.flow_dict['scanner_logits'].cuda())
        # roi priors
        self.roi_base_loc = torch.zeros([self.roi_dim, ], device='cuda', requires_grad=False)
        self.roi_base_scale = torch.ones([self.roi_dim, ], device='cuda', requires_grad=False)
        self.register_buffer('roi_flow_lognorm_loc', torch.zeros([], requires_grad=False))
        self.register_buffer('roi_flow_lognorm_scale', torch.ones([], requires_grad=False))

        # age flows
        self.age_flow_component = Spline(self.age_dim, count_bins=self.spline_bins)
        self.age_flow_lognorm_loc = self.flow_dict['age_mean'].cuda()
        self.age_flow_lognorm_scale = self.flow_dict['age_std'].cuda()
        self.age_flow_normalize = AffineTransform(loc=self.age_flow_lognorm_loc.item(), scale=self.age_flow_lognorm_scale.item())
        self.age_flow_constraint = ComposeTransform([self.age_flow_normalize, ExpTransform()])
        self.age_flow_transforms = ComposeTransform([self.age_flow_component, self.age_flow_constraint])
        # roi flows
        if self.flow_type == 'affine':
            roi_net = DenseNN(self.context_dim, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
            self.roi_flow_component = ConditionalAffineTransform(context_nn=roi_net, event_dim=0)
        elif self.flow_type == 'spline':
            self.roi_flow_component = conditional_spline(self.roi_dim, context_dim=self.context_dim, count_bins=self.spline_bins, order=self.spline_order)
        elif self.flow_type == 'autoregressive':
            self.roi_flow_component = conditional_spline_autoregressive(self.roi_dim, context_dim=self.context_dim, hidden_dims=self.spline_hidden_dims, count_bins=self.spline_bins, order=self.spline_order)
        if self.normalize:
            self.roi_flow_lognorm_loc = self.flow_dict['roi_mean'].cuda()
            self.roi_flow_lognorm_scale = self.flow_dict['roi_std'].cuda()
            self.roi_flow_normalize = AffineTransform(loc=self.roi_flow_lognorm_loc.item(), scale=self.roi_flow_lognorm_scale.item())
            self.roi_flow_constraint = ComposeTransform([self.roi_flow_normalize, ExpTransform()])
            self.roi_flow_transforms = [self.roi_flow_component, self.roi_flow_constraint]
        else:
            self.roi_flow_transforms = ComposeTransformModule([self.roi_flow_component])

    def pgm_model(self):
        # sex
        self.sex_dist = Categorical(logits=self.sex_logits).to_event(1)
        self.sex = pyro.sample('sex', self.sex_dist)
        # age
        self.age_base_dist = Normal(self.age_base_loc, self.age_base_scale).to_event(1)
        self.age_dist = TransformedDistribution(self.age_base_dist, self.age_flow_transforms)
        self.age = pyro.sample('age', self.age_dist)
        age_ = self.age_flow_constraint.inv(self.age)
        # scanner
        self.scanner_dist = Categorical(logits=self.scanner_logits).to_event(1)
        self.scanner = pyro.sample('scanner', self.scanner_dist)
        # roi
        context = torch.cat([self.sex, age_, self.scanner], -1)
        self.roi_base_dist = Normal(self.roi_base_loc, self.roi_base_scale).to_event(1)
        self.roi_dist = ConditionalTransformedDistribution(self.roi_base_dist, self.roi_flow_transforms).condition(context)
        self.roi = pyro.sample('roi', self.roi_dist)

    def forward(self, roi, sex, age, scanner):
        self.pgm_model()
        sex_logp = self.sex_dist.log_prob(sex)
        age_logp = self.age_dist.log_prob(age)
        scanner_logp = self.scanner_dist.log_prob(scanner)
        roi_logp = self.roi_dist.log_prob(roi)
        return {'sex': sex_logp, 'age': age_logp, 'scanner': scanner_logp, 'roi': roi_logp}

    def clear(self):
        self.age_dist.clear_cache()
        self.roi_dist.clear_cache()

    def model(self):
        self.pgm_model()
        return self.sex, self.age, self.scanner, self.roi

    def scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg['fn'], TransformedDistribution):
                return TransformReparam()
            else:
                return None
        return pyro.poutine.reparam(self.model, config=config)(*args, **kwargs)

    def sample(self, n_samples=1):
        with pyro.plate('observations', n_samples):
            samples = self.model()
        return (*samples,)

    def sample_scm(self, n_samples=1):
        with pyro.plate('observations', n_samples):
            samples = self.scm()
        return (*samples,)

    def infer_exogeneous(self, **obs):
        cond_sample = pyro.condition(self.sample, data=obs)
        cond_trace = pyro.poutine.trace(cond_sample).get_trace(obs['roi'].shape[0])

        output = {}
        for name, node in cond_trace.nodes.items():
            if 'fn' not in node.keys():
                continue
            fn = node['fn']
            if isinstance(fn, Independent):
                fn = fn.base_dist
            if isinstance(fn, TransformedDistribution):
                output[name + '_base'] = ComposeTransform(fn.transforms).inv(node['value'])
        return output

    def counterfactual(self, obs, condition, num_particles=1):
        counterfactuals = []
        for _ in range(num_particles):
            exogeneous = self.infer_exogeneous(**obs)
            if 'sex' not in condition.keys():
                exogeneous['sex'] = obs['sex']
            if 'scanner' not in condition.keys():
                exogeneous['scanner'] = obs['scanner']

            counter = pyro.poutine.do(pyro.poutine.condition(self.sample_scm, data=exogeneous), data=condition)(obs['roi'].shape[0])
            counterfactuals += [counter]
        return {k: v for k, v in zip(('sex', 'age', 'scanner', 'roi'), (torch.stack(c).mean(0) for c in zip(*counterfactuals)))}


def _conditionalscm(arch, flow_dict, scanner_dim, flow_type, bins, order):
    model = ConditionalSCM(flow_dict=flow_dict, scanner_dim=scanner_dim,
        flow_type=flow_type, spline_bins=bins, spline_order=order)
    return model

def conditionalscm(flow_dict=None, scanner_dim=6, flow_type='affine', bins=8, order='linear'):
    return _conditionalscm('conditionalscm', flow_dict, scanner_dim, flow_type, bins, order)
