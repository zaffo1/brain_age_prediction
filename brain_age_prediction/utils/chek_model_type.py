'''
Function to check if the 'model_type' used in many funcions
has an accepted value. In particular, it should be a string, and can only be:
'structural', 'functional' or 'joint'.
'''
import sys


def check_model_type(model_type):
    '''
    Check if 'model_type' is a string.
    Check if 'model_type' is either 'structural', 'functional' or 'joint'
    The input 'model_type' should satisfy both these requirements, otherwise,
    terminate the program and return an exit code of 1 to the operating system.
    '''
    if not isinstance(model_type,str):
        print('model_type should be a string!')
        sys.exit(1)

    if model_type not in ('structural','functional','joint'):
        print(f'\'{model_type}\' is not a valid model_type: '
               'it should be either \'structural\' or \'functional\' or \'joint\'')
        sys.exit(1)
