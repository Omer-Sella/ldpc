# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:59:39 2021

@author: Omer Sella
"""
import os
import utilityFunctions


projectDir = os.environ.get('LDPC')
if projectDir == None:
    import pathlib
    projectDir = pathlib.Path(__file__).parent.absolute()


import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, projectDir)

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

def explainLogger():
    keys = ['minimum', 'maximum', 'someOtherValue']
    logPath = projectDir + "/temp/gettingStarted/experiments/"
    myLog = utilityFunctions.logger(keys, logPath)
    print("Logger explained:")
    print("The logger is a class with functions like logPrint, keyValue, dumpLogger.")
    print("Initialise using: myLog = utilityFunctions.logger(keys)")
    print("In this example the keys were set to: " + str(keys))
    print("To print a coloured message use myLog.logPrint(message, colour). Here is an example:")
    myLog.logPrint("An example", 'green')

if __name__ == '__main__':
    status = testEnvironmentVariable()
    print(status)