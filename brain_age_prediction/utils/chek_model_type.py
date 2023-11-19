'''
Function to check if the 'model_type' used in many funcions
has an accepted value. In particular, it should be a string that can only be:
'structural', 'functional' or 'joint'.
'''

def check_model_type(model_type):
    '''
    Check if 'model_type' is either 'structural', 'functional' or 'joint'
    If not, raise an AssertionError

        Parameters:
                    model_type (string): the string indicating the model type
    '''

    assert model_type in ('structural','functional','joint'),(
        f"'{model_type}' is not a valid model_type: "
        "'model_type' must be either 'structural', 'functional' or 'joint'")
