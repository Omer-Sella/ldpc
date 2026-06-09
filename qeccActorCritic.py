import numpy as np
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch import no_grad
import torch
"""
qeccActorCritic
    Is a top module that uses:
    -- multi layer perceptron (explicitMLP)
    For the valuation, since the valuation is a simple function.
    It also uses
    -- bernoulliActor:
    for the policy. The policy is (usually, or in the future) a more complicated function, using potentially several:
        -- multi layer perceptron (explicitMLP)
"""



class explicitMLP(nn.Module):
    """
    explicitMLP creates a multi layer perceptron with explicit input and output lengths.
    if hiddenLayersLengths is not an empty list it will create hidden layers with the specified lengths as input lengths.
    default activation is the identity.
    """
    def __init__(self, firstLayerSize, lastLayerSize, hiddenLayersSpecification, intermediateActivation = nn.Identity, outputActivation = nn.Identity):
        super().__init__()
        lengths = [firstLayerSize] + hiddenLayersSpecification + [lastLayerSize]

        self.outputActivation = outputActivation
        
        self.outputDimension = lastLayerSize
        layerList = []
        
        for l in range(len(lengths) - 1):
            if (l < (len(lengths) - 2)):
                activation = intermediateActivation
            else:
                activation = outputActivation
            layerList = layerList + [nn.Linear(lengths[l], lengths[l + 1]), activation()]
        self.layers = nn.ModuleList(layerList)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation = nn.Identity):
        super().__init__()
        self.v_net = explicitMLP(firstLayerSize=obs_dim, lastLayerSize=1, hiddenLayersSpecification=hidden_sizes, intermediateActivation=activation, outputActivation=nn.Identity)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class Actor(nn.Module):

    def distribution(self, observations):
        raise NotImplementedError

    def logProbabilityFromDistribution(self, policy, actions):
        raise NotImplementedError

    def forward(self, observations, actions=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        policy = self.distribution(observations)
        logp_a = None
        if actions is not None:
            logp_a = self.logProbabilityFromDistribution(policy, actions)
        return policy, logp_a
    
class bernouliActor(Actor):
    def __init__(self, firstLayerSize, lastLayerSize, hiddenLayersSpecification, intermediateActivation = nn.Identity, outputActivation = nn.Identity):
        super().__init__()
        self.logitsNetwork = explicitMLP(firstLayerSize, lastLayerSize, hiddenLayersSpecification)
    
    def distribution(self, observations):
        logits = self.logitsNetwork(observations)
        return Bernoulli(logits = logits)
    
    def logProbabilityFromDistribution(self, policy, actions):
        return policy.log_prob(actions)

    # def forward(self, observations, actions = None):
    #     # Get a parametrized Bernouli distribution, which is a function of observations
    #     policy = self.distribution(observations)
    #     if actions is not None:
    #         logProbabilitiesActions = self.logProbabilityFromDistribution(policy, actions)
    #     else:
    #         logProbabilitiesActions= None
    #     return policy, logProbabilitiesActions
    
class qeccActorCritic(nn.Module):
    """
    A qecc actor critic wraps a policy (actor) and a valuation (critic) together.
    It provides a step function that accepts an observation, and returns an action and an expected value (reward)


    """
    def __init__(self, observationSpaceType, observationSpaceSize, actionSpaceType, actionSpaceSize, policyHiddenLayers, valuationHiddenLayers, actorCriticDevice = 'cpu'):
        super().__init__()
        # Initialize a policy 
        if observationSpaceType == bool:
            self.policy = bernouliActor(observationSpaceSize, actionSpaceSize, policyHiddenLayers)
        else:
            print(f"Actor that returns a non binary action is not implemented yet.")
            raise NotImplementedError
        # Initialize a valuation, the output of a valuation is size 1, i.e. a real scalar
        self.valuation  = MLPCritic(observationSpaceSize, valuationHiddenLayers)
        self.actionSpaceType = actionSpaceType

    def step(self, observations):
        with no_grad():
            # Get an instance of a parametrized Bernouli distribution (which is a function of observations)
            parametrizedBernouliDistribution = self.policy.distribution(observations)
            # Sample from the parametrized distribution
            actions = parametrizedBernouliDistribution.sample()
            # Omer: There is a potential issue here, as the forward of "policy" here returns both the distribution and the log probabilities, but here we only use the distribution. This is not necessarily a problem, but it is a bit inelegant. We can consider changing the forward of "policy" to only return the distribution as currently implemented.
            _, logProbabilityAction =  self.policy(observations, actions)
            values = self.valuation(observations)
        return actions, values, logProbabilityAction
