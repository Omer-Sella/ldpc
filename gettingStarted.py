# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:59:39 2021

@author: Omer Sella
"""
import os

GREEN =  "\033[32m" # green text
WHITE =  "\033[37m" # white text
RED   =  "\033[31m" # red text
YELLOW = "\033[33m" # yellow text
ENDC  = "\033[m"    # reset to default
    
def testEnvironmentVariable():
    status = 'No environment variable'
    environmentVariableExists = (os.environ.get('LDPC') != None)
    if environmentVariableExists:
        status = 'OK'
        print("*****")
    else:
        print(RED + "Variable environment " + GREEN + "LDPC" + RED + " which is a path to the project top directory cannot be found." + ENDC)
        isWindows = (os.name == 'nt')
        if isWindows:
            print(YELLOW + "To set up this variable in Windows 10 search for 'environment variables'. Open 'Edit the system environment variables'. Click 'Environment Variables' and add a new user variable called LDPC with value equal to the project top directory." + ENDC)
        else:
            print(YELLOW + "To set up this variable in linux type 'export LDPC=<projectPath>' into the console, or add it to the shell to make it persistant, where <projectPath> is the path to the project top directory." + ENDC)
    
    return status

if __name__ == '__main__':
    status = testEnvironmentVariable()
    print(status)