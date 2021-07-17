# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:19:52 2021

@author: Omer Sella

Here is the spec from openAI ppo:
 actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================



"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

MODELS_BOOLEAN_TYPE = np.bool
MODELS_INTEGER_TYPE = np.int32
CIRCULANT_SIZE = 511


def numToBits(number, numberOfBits):
    assert number < 16
    assert number >= 0
    newNumber = np.zeros(numberOfBits, dtype = int)
    for j in range(numberOfBits - 1, -1, -1):
        newNumber[j] = newNumber[j] + (number % 2)
        number = number >> 1
    return newNumber

class explicitMLP(nn.Module):
    """
    explicitMLP creates a multi layer perceptron with explicit input and output lengths.
    if hiddenLayersLengths is not an empty list it will create hidden layers with the specified lengths as input lengths.
    default activation is the identity.
    """
    def __init__(self, inputLength, outputLength, hiddenLayersLengths, intermediateActivation = nn.Identity, outputActivation = nn.Identity):
        super().__init__()
        lengths = [inputLength] + hiddenLayersLengths + [outputLength]
        self.outputActivation = outputActivation
        layerList = []
        
        for l in range(len(lengths) - 1):
            if (l < (len(lengths) - 2)):
                activation = intermediateActivation
            else:
                activation = outputActivation
            layerList = layerList + [nn.Linear(lengths[l], lengths[l + 1]), activation()]
        self.layers = nn.ModuleList(layerList)
        self.outputDimension = outputLength
        
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
   
    
class actorCritic(nn.Module):
    def __init__(self, observationSpaceType, observationSpaceSize, actionSpaceType, 
                 actionSpaceSize, hiddenEncoderSize, maximumNumberOfHotBits, hiddenLayerParameters, actorCriticDevice = 'cpu'):
        super().__init__()
        self.observationSpaceType = observationSpaceType
        self.observationSpaceSize = observationSpaceSize
        self.actionSpaceType = actionSpaceType
        self.actionSpaceSize = actionSpaceSize
        self.device = actorCriticDevice
        self.maximumNumberOfHotBits = maximumNumberOfHotBits
        self.hiddenEncoderSize = hiddenEncoderSize
        
        self.rowCoordinateRange = 2
        self.columnCoordinateRange = 16
        self.circulantSize = 511
        self.defaultHiddenLayerSizes = [64]
        self.defaultActivation = nn.Identity

        self.encoder = explicitMLP(observationSpaceSize, hiddenEncoderSize, [hiddenEncoderSize, hiddenEncoderSize])

        self.rowCoordinateModel = explicitMLP(self.hiddenEncoderSize, self.rowCoordinateRange, self.defaultHiddenLayerSizes)

        self.columnCoordinateModel = explicitMLP(self.hiddenEncoderSize + 1, self.columnCoordinateRange, self.defaultHiddenLayerSizes)
        
        self.numberOfHotBitsModel = explicitMLP(self.hiddenEncoderSize + 2, self.maximumNumberOfHotBits, self.defaultHiddenLayerSizes)
        
        self.kHotVectorGenerator = explicitMLP(self.circulantSize, self.circulantSize, self.defaultHiddenLayerSizes)
        
        self.encoder2 = explicitMLP(self.hiddenEncoderSize + 3, self.circulantSize, self.defaultHiddenLayerSizes)
        
        ## Omer: A note to self: the critic valuates the present state, not the state you are going to be in after taking the action.
        self.critic = explicitMLP(self.hiddenEncoderSize, 1, [hiddenEncoderSize, hiddenEncoderSize])
        self.to(actorCriticDevice)
    
    
    
    def actorActionToEnvAction(self, actorAction):
        i, j, k, hotCoordinates = actorAction
        """
        The actor is expected to produce i, j, and up to k coordinates which will be hot.
        The environment is expecting i,j and a binary vector.
        """
        binaryVector = np.zeros(self.circulantSize, dtype = MODELS_BOOLEAN_TYPE)
        binaryVector[hotCoordinates[0:k]] = 1
        environmentStyleAction = [i, j, binaryVector]
        return environmentStyleAction
    
    def step(self, observations, action = None):
        
        # The step function has 3 modes:
        #   1. Training - this is where we use the model and sample from the distributions parametrised by the model.
        #   2. Most probable action - this is where we use the deterministic model and instead of sampling take the most probable action.
        #   3. Both observations AND actions are provided, in which case we are evaluating log probabilities
        
        # Observations batchSize X observationSpaceSize of type observationSpaceType

        if action is not None:
            action = torch.as_tensor(action, device = self.device)
        
        encodedObservation = self.encoder(observations)
        logitsForIChooser = self.rowCoordinateModel(encodedObservation)
        iCategoricalDistribution = Categorical(logits = logitsForIChooser)
        
        if action is not None:
            i = action[:, 0]
        elif self.training:
            i = iCategoricalDistribution.sample()
        else:
            i = torch.argmax(logitsForIChooser)
               
        
                
        # Omer Sella: now we need to append i to the observations
        ## Omer Sella: when acting you need to sample and concat. When evaluating, you need to break the action into internal components and set to them.
        ## Then log probabilities are evaluated at the end (regardless of whether this was sampled or given)
        
        i = i.float()
        iTensor = i.unsqueeze(0)
        iAppendedObservations = torch.cat([encodedObservation, iTensor], dim = -1)
        logitsForJChooser = self.columnCoordinateModel(iAppendedObservations)
        jCategoricalDistribution = Categorical(logits = logitsForJChooser)
        
        if action is not None:
            j = action[:, 1]
        elif self.training:    
            j = jCategoricalDistribution.sample()
        else:
            j = torch.argmax(logitsForJChooser)
                
        # Omer Sella: now we need to append j to the observations
        j = j.float()
        jTensor = j.unsqueeze(0)
        jAppendedObservations = torch.cat([iAppendedObservations, jTensor], dim = -1)
        logitsForKChooser = self.numberOfHotBitsModel(jAppendedObservations)
        kCategoricalDistribution = Categorical(logits = logitsForKChooser)
        
        if action is not None:
            k = action[:, 2]
        elif self.training:
            k = kCategoricalDistribution.sample()
        else:
            k = torch.argmax(logitsForKChooser)
        
        k = k.float()
        kTensor = k.unsqueeze(0)
        kAppendedObservations = torch.cat([jAppendedObservations, kTensor], dim = -1)
        setEncodedStuff = self.encoder2(kAppendedObservations)
        
        
        logProbCoordinates = np.zeros(self.maximumNumberOfHotBits)
        
        if action is not None:
            coordinates = action[:, 3 : 3 + self.maximumNumberOfHotBits]
            idx = 0
            while idx < k:
                logitsForCoordinateChooser = self.kHotVectorGenerator(setEncodedStuff)
                circulantSizeCategoricalDistribution = Categorical(logits = logitsForCoordinateChooser)
                newCoordinate = coordinates[idx]
                logProbCoordinates[idx] = circulantSizeCategoricalDistribution.log_prob(newCoordinate)
                setEncodedStuff = setEncodedStuff + logitsForCoordinateChooser
                idx = idx + 1
        elif self.training:
            coordinates = -1 * np.ones(self.maximumNumberOfHotBits)
            idx = 0
            while idx < k:
                logitsForCoordinateChooser = self.kHotVectorGenerator(setEncodedStuff)
                circulantSizeCategoricalDistribution = Categorical(logits = logitsForCoordinateChooser)
                newCoordinate = circulantSizeCategoricalDistribution.sample()
                coordinates[idx] = newCoordinate
                logProbCoordinates[idx] = circulantSizeCategoricalDistribution.log_prob(newCoordinate)
                setEncodedStuff = setEncodedStuff + logitsForCoordinateChooser
                idx = idx + 1
        else:
            coordinates = -1 * np.ones(self.maximumNumberOfHotBits)
            idx = 0
            while idx < k:
                logitsForCoordinateChooser = self.kHotVectorGenerator(setEncodedStuff)
                circulantSizeCategoricalDistribution = Categorical(logits = logitsForCoordinateChooser)
                newCoordinate = torch.argmax(logitsForCoordinateChooser)
                coordinates[idx] = newCoordinate
                logProbCoordinates[idx] = circulantSizeCategoricalDistribution.log_prob(newCoordinate)
                setEncodedStuff = setEncodedStuff + logitsForCoordinateChooser
                idx = idx + 1
                
                    
        #log probs
        logpI = iCategoricalDistribution.log_prob(i)
        logpJ = jCategoricalDistribution.log_prob(j).sum(axis = -1)
        logpK = kCategoricalDistribution.log_prob(k).sum(axis = -1)
        #logProbabilityList = [,
        #    ,
        #    ]
        
                
                
        return i, j, k.item(), coordinates, logpI, logpJ, logpK, logProbCoordinates



class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act = None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        raise NotImplementedError

class openAIActor(Actor):
        def __init__(self, observationSpaceType, observationSpaceSize, actionSpaceType, actionSpaceSize, hiddenEncoderSize, maximumNumberOfHotBits, hiddenLayerParameters, actorCriticDevice = 'cpu'):
            super().__init__()
            self.observationSpaceType = observationSpaceType
            self.observationSpaceSize = observationSpaceSize
            self.actionSpaceType = actionSpaceType
            self.actionSpaceSize = actionSpaceSize
            self.device = actorCriticDevice
            self.maximumNumberOfHotBits = maximumNumberOfHotBits
            self.hiddenEncoderSize = hiddenEncoderSize
            self.rowCoordinateRange = 2
            self.columnCoordinateRange = 16
            self.circulantSize = 511
            self.defaultHiddenLayerSizes = [64]
            self.defaultActivation = nn.Identity
            self.encoder = explicitMLP(observationSpaceSize, hiddenEncoderSize, [hiddenEncoderSize, hiddenEncoderSize])
            self.rowCoordinateModel = explicitMLP(self.hiddenEncoderSize, self.rowCoordinateRange, self.defaultHiddenLayerSizes)
            self.columnCoordinateModel = explicitMLP(self.hiddenEncoderSize + 1, self.columnCoordinateRange, self.defaultHiddenLayerSizes)
            self.numberOfHotBitsModel = explicitMLP(self.hiddenEncoderSize + 2, self.maximumNumberOfHotBits, self.defaultHiddenLayerSizes)
            self.kHotVectorGenerator = explicitMLP(self.circulantSize, self.circulantSize, self.defaultHiddenLayerSizes)
            self.encoder2 = explicitMLP(self.hiddenEncoderSize + 3, self.circulantSize, self.defaultHiddenLayerSizes)
            self.to(actorCriticDevice)
        
        def actorActionToEnvAction(self, actorAction):
            i, j, k, hotCoordinates = actorAction
            """
            The actor is expected to produce i, j, and up to k coordinates which will be hot.
            The environment is expecting i,j and a binary vector.
            """
            binaryVector = np.zeros(self.circulantSize, dtype = MODELS_BOOLEAN_TYPE)
            binaryVector[hotCoordinates[0:k]] = 1
            environmentStyleAction = [i, j, binaryVector]
            return environmentStyleAction
    
        def step(self, observations, action = None):
        
            # The step function has 3 modes:
            #   1. Training - this is where we use the model and sample from the distributions parametrised by the model.
            #   2. Most probable action - this is where we use the deterministic model and instead of sampling take the most probable action.
            #   3. Both observations AND actions are provided, in which case we are evaluating log probabilities
        
            # Observations batchSize X observationSpaceSize of type observationSpaceType

            if action is not None:
                action = torch.as_tensor(action, device = self.device)
        
            encodedObservation = self.encoder(observations)
            logitsForIChooser = self.rowCoordinateModel(encodedObservation)
            iCategoricalDistribution = Categorical(logits = logitsForIChooser)
            iDistributionEntropy = iCategoricalDistribution.entropy().unsqueeze(-1)
            if action is not None:
                i = action[:, 0]
            elif self.training:
                i = iCategoricalDistribution.sample()
            else:
                i = torch.argmax(logitsForIChooser)
               
        
                
            # Omer Sella: now we need to append i to the observations
            ## Omer Sella: when acting you need to sample and concat. When evaluating, you need to break the action into internal components and set to them.
            ## Then log probabilities are evaluated at the end (regardless of whether this was sampled or given)
        
            i = i.float()
            iTensor = i.unsqueeze(-1)
            iAppendedObservations = torch.cat([encodedObservation, iTensor], dim = -1)
            logitsForJChooser = self.columnCoordinateModel(iAppendedObservations)
            jCategoricalDistribution = Categorical(logits = logitsForJChooser)
            jDistributionEntropy = jCategoricalDistribution.entropy().unsqueeze(-1)

            if action is not None:
                j = action[:, 1]
            elif self.training:    
                j = jCategoricalDistribution.sample()
            else:
                j = torch.argmax(logitsForJChooser)
                
            # Omer Sella: now we need to append j to the observations
            j = j.float()
            jTensor = j.unsqueeze(-1)
            jAppendedObservations = torch.cat([iAppendedObservations, jTensor], dim = -1)
            logitsForKChooser = self.numberOfHotBitsModel(jAppendedObservations)
            kCategoricalDistribution = Categorical(logits = logitsForKChooser)
            kDistributionEntropy = kCategoricalDistribution.entropy().unsqueeze(-1)
        
            if action is not None:
                k = action[:, 2]
            elif self.training:
                k = kCategoricalDistribution.sample()
                #Omer Sella: k can't be 0
                k = kCategoricalDistribution.sample() + 1
            else:
                #k = torch.argmax(logitsForKChooser)
                #Omer Sella: k can't be 0
                k = torch.argmax(logitsForKChooser) + 1
        
            k = k.float()
            kTensor = k.unsqueeze(-1)
            kAppendedObservations = torch.cat([jAppendedObservations, kTensor], dim = -1)
            setEncodedStuff = self.encoder2(kAppendedObservations)
        
        
            
            
        
            # In this part we choose k coordinates, where: k <= maximumNumberOfHotBits <= circulantSize 
            # In practice we choose maximumNumberOfHotBits coordinates, and use only the first k of them
            
            if action is not None:
                coordinates = action[:, 3 : 3 + self.maximumNumberOfHotBits]
                numberOfObservations = coordinates.shape[0]
                print(coordinates.shape[0])
                coordinateEntropies = torch.zeros((numberOfObservations, self.maximumNumberOfHotBits))
                logProbCoordinates = torch.zeros((numberOfObservations, self.maximumNumberOfHotBits))
                
                
                idx = 0
                while idx < self.maximumNumberOfHotBits:
                    logitsForCoordinateChooser = self.kHotVectorGenerator(setEncodedStuff)
                    circulantSizeCategoricalDistribution = Categorical(logits = logitsForCoordinateChooser)
                    newCoordinate = coordinates[:, idx]
                    logProbCoordinates[:, idx] = circulantSizeCategoricalDistribution.log_prob(newCoordinate)
                    coordinateEntropies[:, idx] = circulantSizeCategoricalDistribution.entropy()# Omer Sella: commented this: .unsqueeze(-1)
                    setEncodedStuff = setEncodedStuff + logitsForCoordinateChooser
                    idx = idx + 1
            elif self.training:
                coordinateEntropies = torch.zeros(self.maximumNumberOfHotBits)
                logProbCoordinates = torch.zeros(self.maximumNumberOfHotBits)
                coordinates = -1 * np.ones(self.maximumNumberOfHotBits)
                idx = 0
                while idx < self.maximumNumberOfHotBits:
                    logitsForCoordinateChooser = self.kHotVectorGenerator(setEncodedStuff)
                    circulantSizeCategoricalDistribution = Categorical(logits = logitsForCoordinateChooser)
                    newCoordinate = circulantSizeCategoricalDistribution.sample()
                    coordinates[idx] = newCoordinate
                    logProbCoordinates[idx] = circulantSizeCategoricalDistribution.log_prob(newCoordinate)
                    coordinateEntropies[idx] = circulantSizeCategoricalDistribution.entropy().unsqueeze(-1)
                    setEncodedStuff = setEncodedStuff + logitsForCoordinateChooser
                    idx = idx + 1
            else:
                coordinateEntropies = torch.zeros(self.maximumNumberOfHotBits)
                logProbCoordinates = torch.zeros(self.maximumNumberOfHotBits)
                coordinates = -1 * np.ones(self.maximumNumberOfHotBits)
                idx = 0
                while idx < self.maximumNumberOfHotBits:
                    logitsForCoordinateChooser = self.kHotVectorGenerator(setEncodedStuff)
                    circulantSizeCategoricalDistribution = Categorical(logits = logitsForCoordinateChooser)
                    newCoordinate = torch.argmax(logitsForCoordinateChooser)
                    coordinates[idx] = newCoordinate
                    logProbCoordinates[idx] = circulantSizeCategoricalDistribution.log_prob(newCoordinate)
                    coordinateEntropies[idx] = circulantSizeCategoricalDistribution.entropy().unsqueeze(-1)
                    setEncodedStuff = setEncodedStuff + logitsForCoordinateChooser
                    idx = idx + 1
                
                    
            #log probs
            logpI = iCategoricalDistribution.log_prob(i).unsqueeze(-1)
            logpJ = jCategoricalDistribution.log_prob(j).unsqueeze(-1)#.sum(axis = -1)
            #Omer Sella: remember that you added 1 to k so to get the log prob reduce 1
            logpK = kCategoricalDistribution.log_prob(k-1).unsqueeze(-1)#.sum(axis = -1)
            
            
            if action is None:
                i = np.int32(i.item())
                j = np.int32(j.item())
                k = np.int32(k.item())
                coordinates = np.int32(coordinates)
                
                
            return i, j, k, coordinates, logpI, logpJ, logpK, logProbCoordinates, iDistributionEntropy, jDistributionEntropy, kDistributionEntropy, coordinateEntropies


class openAIActorCritic(nn.Module):

    def __init__(self, observationSpaceType, observationSpaceSize, actionSpaceType, actionSpaceSize, hiddenEncoderSize, maximumNumberOfHotBits, hiddenLayerParameters, actorCriticDevice = 'cpu'):
    
        super().__init__()
        self.pi = openAIActor(observationSpaceType, observationSpaceSize, actionSpaceType, actionSpaceSize, hiddenEncoderSize, maximumNumberOfHotBits, hiddenLayerParameters, actorCriticDevice)
        self.v  = explicitMLP(observationSpaceSize, 1, [hiddenEncoderSize, hiddenEncoderSize])

    #def step(self, obs):
    #    with torch.no_grad():
    #        pi = self.pi._distribution(obs)
    #        a = pi.sample()
    #        logp_a = self.pi._log_prob_from_distribution(pi, a)
    #        v = self.v(obs)
    #        #print("*** actor_critic step debug***")
    #        #print("*** action:")
    #        #print(a)
    #        #print(a.shape)
    #        a = a.numpy().astype(int)
    #        v = v.numpy()
    #        logp_a = logp_a.numpy()
    #    return a, v, logp_a

    #def act(self, obs):
    #    return self.step(obs)[0]

    def step(self, obs, actions = None):
        i, j, k, coordinates, logpI, logpJ, logpK, logProbCoordinates, iDistributionEntropy, jDistributionEntropy, kDistributionEntropy, coordinateEntropies = self.pi.step(obs, actions)
        v = self.v(obs)
        vector = np.zeros(CIRCULANT_SIZE, dtype = np.int32)
        #print(k)
        #print(coordinates)
        #print(coordinates[0: np.int(k)])
        # Omer Sella: if action is None, then k is an integer >= 0 and all is ok to use it as an index.
        # If action is not None, then this "step" function is just supposed to evaulate entropies and log probabilities, not produce new actions, so
        # it's ok to return the environment vector action as zeros.
        if actions is None:
            vector[coordinates[0: k]] = 1
            xCoordinate = numToBits(i, 1)
            yCoordinate = numToBits(j, 4)
            ppoBufferAction = np.hstack(([i,j,k], coordinates))
            envAction = np.hstack((np.hstack((xCoordinate, yCoordinate)), vector))
        else:
            # Omer Sella: temporary fix for when action is not none:
            xCoordinate = numToBits(1, 1)
            yCoordinate = numToBits(1, 1)
            ppoBufferAction = False #np.hstack(([i,j,k], coordinates))
            envAction = False#np.hstack((np.hstack((xCoordinate, yCoordinate)), vector))
        logp_list = [logpI,  logpJ, logpK, logProbCoordinates]
        #print(logp_list)
        #for l in logp_list:
        #    print(l)
        logp = torch.cat(logp_list, dim = -1)
        #print(logp)
        logPSummed = logp.sum(dim = -1, keepdim = False)
        #print(logPSummed)
        #print(ppoBufferAction)
        entropy_list = [iDistributionEntropy, jDistributionEntropy, kDistributionEntropy, coordinateEntropies]
        entropy = torch.cat(entropy_list, dim = -1)
        entropySummed = entropy.sum(dim = -1, keepdim = False)
        a = [i, j, k, coordinates, ppoBufferAction, envAction]
        return a, v.detach().numpy(), logPSummed, entropySummed, logp_list, entropy_list

    
def testExplicitMLP():
    inputLength = 2048
    outputLength = 16
    hiddenLayersLengths = [64, 64]
    myMLP = explicitMLP(inputLength, outputLength, hiddenLayersLengths)
    return myMLP

def testExplicitMLPForward():
    myMLP = testExplicitMLP()
    testVector = torch.rand(2048)
    y = myMLP(testVector)
    return y


def testActorCritic():
     testAC = actorCritic(observationSpaceType = int, observationSpaceSize = 2048, actionSpaceType = int, actionSpaceSize = (1 + 1 + 1 + 7), hiddenEncoderSize = 64, maximumNumberOfHotBits = 7, hiddenLayerParameters = [64,64], actorCriticDevice = 'cpu')
     print(testAC.parameters())
     testVector = torch.rand(2048)
     print(testVector.size())
     result = testAC.step(torch.rand(2048))
     print(result)
     return 'OK'

if __name__ == '__main__':
    print("***You hit play on the wrong file #**hole... modulu ...")
    testActorCritic()
