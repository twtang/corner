from fastai.tabular.all import *

import torch
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, probs_to_logits, lazy_property, logits_to_probs

    
# Reference: [1] An alternative bivariate negative binomial model based on
#                Sarmanov family (Lee, 2021) and
#            [2] On the bivariate negative binomial regression model (Famoye, 2010)
# Denote as BNBSL model.
class BivariateNegativeBinomialSL(Distribution):
    r"""
    Creates a Bivariate Negative Binomial Sarmanov-Lee distribution, i.e. distribution
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
    arg_constraints = {'total_count': constraints.independent(constraints.greater_than_eq(0), 1),
                       'probs': constraints.independent(constraints.half_open_interval(0., 1.), 1),
                       'logits': constraints.real_vector}
    support = constraints.independent(constraints.nonnegative_integer, 1)

    def __init__(self, total_count, omega, probs=None, logits=None, validate_args=None):
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
#             self.total_count, self.probs, = broadcast_all(total_count, probs)
            self.total_count, self.probs, self.omega = total_count, probs, omega
            self.total_count = self.total_count.type_as(self.probs)
        else:
#             self.total_count, self.logits, = broadcast_all(total_count, logits)
            self.total_count, self.logits, self.omega = total_count, logits, omega
            self.total_count = self.total_count.type_as(self.logits)
        
        self._param = self.probs if probs is not None else self.logits
        batch_shape = self._param.size()
        super(BivariateNegativeBinomialSL, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(BivariateNegativeBinomialSL, _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        new.omega = self.omega.expand(batch_shape)
        if 'probs' in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if 'logits' in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(BivariateNegativeBinomialSL, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @property
    def mean(self):
        return self.total_count * torch.exp(self.logits)

    @property
    def variance(self):
        return self.mean / torch.sigmoid(-self.logits)

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs)

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits)

    @property
    def alpha(self):
        return 1. / self.total_count
    
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

        # Method 1 (Lee, 2021):        
        Q = (((1. + self.alpha * self.mean) / (1. + self.alpha + self.alpha * self.mean)) ** (value + self.total_count) - 
             1. / (1. + self.alpha) ** self.total_count)
        log_copula = torch.log(torch.clamp(1. + self.omega * Q.prod(-1), 1e-8))

        # Method 2 (Famoye, 2010):
        # d = math.exp(-1.)
        # c = (1. + d * self.alpha * self.mean) ** (-self.total_count)
        # log_copula = torch.log(torch.clamp(1. + self.omega * (torch.exp(-value) - c).prod(-1), 1e-8))

        log_unnormalized_prob = (self.total_count * F.logsigmoid(-self.logits) +
                                 value * F.logsigmoid(self.logits)).sum(-1)
        
        log_normalization = (-torch.lgamma(self.total_count + value) + torch.lgamma(1. + value) +
                             torch.lgamma(self.total_count)).sum(-1)

        return log_unnormalized_prob - log_normalization + log_copula

    
class BivariateNegativeBinomialSLNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, parameters, target):
        # Time-weighted
        if target.size(-1) == 3:
            value, weight = target[:, :2], target[:, 2]
        else:
            value, weight = target, None
            
        alpha = F.softplus(parameters[:, 0:2])
        mu = F.softplus(parameters[:, 2:4])
        omega = parameters[:, 4]
        # omega = torch.tanh(parameters[:, 4]) * 50.

        total_count = 1. / alpha
        logits = torch.log(alpha * mu)

        # target: number of successes before `total_count` number of failures
        # total_count: number of failures
        # probs: prob. of success
        # total_count > 0
        distribution = BivariateNegativeBinomialSL(total_count=total_count, omega=omega, logits=logits)
        likelihood = distribution.log_prob(value)
        if weight is not None: likelihood *= weight
        return -likelihood.mean()
    

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