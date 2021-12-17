# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:21:31 2019

@author: Omer
"""

from gym.envs.registration import register

register(
    id='ldpc-v0',
    entry_point='gym_ldpc.envs:LdpcEnv',
    kwargs={'replacementOnly' : False, 'seed' : 7134066},
)

