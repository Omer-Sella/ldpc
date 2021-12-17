# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:56:34 2020

@author: Omer
"""

# Omer Sella: this function tests the registration of the ldpc environment to the gym registry.

def testRegistry():
    status = 'FAIL'
    from gym import envs
    allEnvs = envs.registry.all()
    envIDs = [envSpec.id for envSpec in allEnvs]
    if ('ldpc-v0' in envIDs):
        status = 'OK'
    return status