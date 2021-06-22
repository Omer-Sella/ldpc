# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:30:45 2020

@author: Omer
"""

import ldpc
import fileHandler
#import ldpcTF
import ldpcCUDA
import common
import models
import utilityFunctions




def test_modulationAndSlicingRoundTrip():
    assert ldpc.testModulationAndSlicingRoundTrip() == 'OK'

  
def test_fileHandler_nearEarth():
    assert fileHandler.testFileHandler() == 'OK'
    
#def test_testModulationAndSlicingRoundtripTFimplementation():
#    assert ldpcTF.testModulationAndSlicingRoundtrip() == 'OK'
    
#def test_minSumStepTFimplementation():    
#    assert ldpcTF.testMinSumStep == 'OK'

def test_ldpcCUDAdecoder():
    _, status = ldpcCUDA.testNearEarth(graphics = False)
    assert status =='OK'
    
    
def test_models():
    status = models.testActorCritic()
    assert status == 'OK'
    
    
def test_utilityFunctionLogger():
    status = utilityFunctions.testLogger()
    assert status == 'OK'
    
#def test_utilityFunctionPlotter():
#    status = utilityFunctions.testPlotter()
#    assert status == 'OK'

def test_roundtripCompression():
    import ldpc_env
    status = ldpc_env.testCompressionRoundrip()
    assert status == 'OK'