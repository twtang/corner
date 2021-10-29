from fastai.tabular.all import *

import torch
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, probs_to_logits, lazy_property, logits_to_probs

class BivariateNegativeBinomial(Distribution):
    r"""
    Creates a Bivariate Negative Binomial distribution, i.e. distribution
    of the number of successful independent and identical Bernoulli trials
    before :attr:`total_count` failures are achieved. The probability
    of failure of each Bernoulli trial is :attr:`probs`.

    Args:
        total_count (float or Tensor): non-negative number of negative Bernoulli
            trials to stop, although the distribution is still valid for real
            valued count
        probs (Tensor): Event probabilities of failure in the half open interval [0, 1)
        logits (Tensor): Event log-odds for probabilities of failure
    """
    arg_constraints = {'total_count': constraints.greater_than_eq(0),
                       'probs': constraints.simplex,
                       'logits': constraints.real_vector}
#     support = constraints.nonnegative_integer

    def __init__(self, total_count, probs=None, logits=None, validate_args=None):
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
#             self.total_count, self.probs, = broadcast_all(total_count, probs)
            self.total_count, self.probs, = total_count, probs
            self.total_count = self.total_count.type_as(self.probs)
        else:
#             self.total_count, self.logits, = broadcast_all(total_count, logits)
            self.total_count, self.logits, = total_count, logits
            self.total_count = self.total_count.type_as(self.logits)

        self._param = self.probs[..., 1:] if probs is not None else self.logits[..., 1:]
        batch_shape = self._param.size()
        super(BivariateNegativeBinomial, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(BivariateNegativeBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        if 'probs' in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if 'logits' in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(BivariateNegativeBinomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @constraints.dependent_property(is_discrete=True, event_dim=1)
    def support(self):
        total_count = torch.ones_like(self.total_count) * 1e3
        return constraints.multinomial(total_count)
    
    @property
    def mean(self):
        return self.total_count * self.probs / self.probs[0]

    @property
    def variance(self):
        return self.mean * (self.probs[0] + self.probs) / self.probs[0]

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs)


    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits)


    @property
    def param_shape(self):
        return self._param.size()

    @lazy_property
    def _gamma(self):
        # Note we avoid validating because self.total_count can be zero.
        return torch.distributions.Gamma(concentration=self.total_count,
                                         rate=torch.exp(-self.logits),
                                         validate_args=False)

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            rate = self._gamma.sample(sample_shape=sample_shape)
            return torch.poisson(rate)


    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        log_unnormalized_prob = self.total_count * self.logits[..., 0] + (value * self.logits[..., 1:]).sum(-1)

        log_normalization = (-torch.lgamma(self.total_count + value.sum(-1)) + 
                             torch.lgamma(1. + value[..., 0]) + 
                             torch.lgamma(1. + value[..., 1]) + 
                             torch.lgamma(self.total_count))

        return log_unnormalized_prob - log_normalization
    
    
class BivariateNegtiveBinomialNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax_probs = nn.Softmax(dim=-1)
        self.softplus_total_count = nn.Softplus()
        
    def forward(self, parameters, target):
        probs = self.softmax_probs(parameters[:, 0:3])
        alpha = self.softplus_total_count(parameters[:, 3])
        total_count = 1 / alpha
        # total_count = self.softplus_total_count(parameters[:, 3])
        # target: number of successes before `total_count` number of failures
        # total_count: number of failures
        # probs: prob. of success
        # total_count > 0
        # probs: [0, 1], sum(probs) = 1
        distribution = BivariateNegativeBinomial(total_count=total_count, probs=probs)
        likelihood = distribution.log_prob(target)
        return -likelihood.mean()
    
    
# def tabular_learner_bnb(data:DataBunch, layers:Collection[int], emb_szs:Dict[str,int]=None, metrics=None,
#         ps:Collection[float]=None, emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True, **learn_kwargs):
#     "Get a `Learner` using `data`, with `metrics`, including a `TabularModel` created using the remaining params."
#     emb_szs = data.get_emb_szs(ifnone(emb_szs, {}))
#     model = TabularModel(emb_szs, len(data.cont_names), out_sz=4, layers=layers, ps=ps, emb_drop=emb_drop,
#                          y_range=y_range, use_bn=use_bn)
#     return Learner(data, model, metrics=metrics, **learn_kwargs)


## Poisson NLL correction
def _poisson_nll_loss(input, target, log_input=True, full=False, size_average=None, eps=1e-8,
                     reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, bool, bool, Optional[bool], float, Optional[bool], str) -> Tensor
    r"""Poisson negative log likelihood loss.

    See :class:`~torch.nn.PoissonNLLLoss` for details.

    Args:
        input: expectation of underlying Poisson distribution.
        target: random sample :math:`target \sim \text{Poisson}(input)`.
        log_input: if ``True`` the loss is computed as
            :math:`\exp(\text{input}) - \text{target} * \text{input}`, if ``False`` then loss is
            :math:`\text{input} - \text{target} * \log(\text{input}+\text{eps})`. Default: ``True``
        full: whether to compute full loss, i. e. to add the Stirling
            approximation term. Default: ``False``
            :math:`\text{target} * \log(\text{target}) - \text{target} + 0.5 * \log(2 * \pi * \text{target})`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        eps (float, optional): Small value to avoid evaluation of :math:`\log(0)` when
            :attr:`log_input`=``False``. Default: 1e-8
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    """
    
    input = input.flatten()
    
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    if log_input:
        loss = torch.exp(input) - target * input
    else:
        loss = input - target * torch.log(input + eps)
    if full:
        mask = target > 1
        loss[mask] += (target * torch.log(target) - target + 0.5 * torch.log(2 * math.pi * target))[mask]#.unsqueeze(1)
    if reduction == 'none':
        ret = loss
    elif reduction == 'mean':
        ret = torch.mean(loss)
    elif reduction == 'sum':
        ret = torch.sum(loss)
    else:
        ret = input
        raise ValueError(reduction + " is not valid")
    return ret

F.poisson_nll_loss = _poisson_nll_loss

class PoissonNLLLoss(nn.PoissonNLLLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(full=True)