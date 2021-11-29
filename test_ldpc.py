# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:30:45 2020

@author: Omer
"""

def test_binarySpace():
    # This function tests that binarySpace can be properly used. 
    # I don't know why, but from python 3.6.10, when I execute the following lines
    # I get an error saying: "AttributeError: can't set attribute"
    status = 'BAD'
    from binarySpace import binarySpace
    testCase = binarySpace(516)
    if testCase.shape[0] == 516:
        status = 'OK'
    return status
    

def test_roundtripCompression():
    import ldpc_env
    status = ldpc_env.testCompressionRoundrip()
    assert status == 'OK'

def test_ldpcCUDAdecoder():
    import ldpcCUDA
    _, status = ldpcCUDA.testNearEarth(graphics = False)
    assert status =='OK'
import models

def test_fileHandler_nearEarth():
    import fileHandler
    assert fileHandler.testFileHandler() == 'OK'
    
def test_models():
    status = models.testActorCritic()
    assert status == 'OK'
import utilityFunctions    
    
def test_utilityFunctionLogger():
    status = utilityFunctions.testLogger()
    assert status == 'OK'
    


def test_modulationAndSlicingRoundTrip():
    import ldpc
    assert ldpc.testModulationAndSlicingRoundTrip() == 'OK'
    
def test_gym_ldpc_environment_is_able_to_init():
    import gym
    ldpcEnvironment = gym.make('gym_ldpc:ldpc-v0')
    assert ldpcEnvironment.gpuDeviceNumber == 0
    

#import ldpcTF

  

#def test_testModulationAndSlicingRoundtripTFimplementation():
#    assert ldpcTF.testModulationAndSlicingRoundtrip() == 'OK'
    
#def test_minSumStepTFimplementation():    
#    assert ldpcTF.testMinSumStep == 'OK'



    
    

#def test_utilityFunctionPlotter():
#    status = utilityFunctions.testPlotter()
#    assert status == 'OK'

