import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete


import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    #print(" *** mlp constructor debug")
    #print(sizes)
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        #print("*** mlp constructor debug: ")
        #print(sizes[j])
        #print(sizes[j+1])
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    #print("*** Debugging mlp.")
    #print(layers)
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a



# 511 binary vector.

# Omer Sella: temporary hack - I turned MLPCategoricalActor into a Bernouli actor
class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        #print("*** *** mlp init debug:")
        #print(type(obs_dim))
        #print(act_dim)
        #print([obs_dim] + list(hidden_sizes) + [act_dim])
        #print("********************")
        #print(activation)
        #print("********************")
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        #print("***** Debugging the _distribution function.")
        #print("***** input obs to _distribution is : ")
        #print(obs.shape)
        #print(obs)
        logits = self.logits_net(obs)
        #print("*** Now let's look at the output:")
        #print(logits)
        #print(logits.shape)
        # Omer Sella: review with Robert P. :-)
        return Bernoulli(logits = logits)#Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        ## Omer Sella: I replaced:
        #return pi.log_prob(act)
        ## with return pi.log_prob(act).sum(axis = -1) to be like gaussian, but this is not necessarily true / correct.
        return pi.log_prob(act).sum(axis = -1)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)
        ## Omer Sella: 01/12/2020 added a hack to include binarySpace which is almost the same as discrete space. Here 2 replaces action_space.n, because action_space.n is only for discrete and marks how many points in the discrete space
        
        else:
            #print('***action space is not box or discrete, assuming binary: ')
            #print("obs_dim == ")
            #print(obs_dim)
            #print("action_space == ")
            #print(action_space)
            #print("action_space.shape[0] == ")
            #print(action_space.shape[0])
            self.pi = MLPCategoricalActor(obs_dim, 516, hidden_sizes, activation) #Omer Sella: temporary fix, 516 is the action space size.
            #print("*** Great success !")

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
            #print("*** actor_critic step debug***")
            #print("*** action:")
            #print(a)
            #print(a.shape)
            a = a.numpy().astype(int)
            v = v.numpy()
            logp_a = logp_a.numpy()
        return a, v, logp_a

    def act(self, obs):
        return self.step(obs)[0]