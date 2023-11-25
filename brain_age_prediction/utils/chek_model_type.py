'''
Function to check if the 'model_type' used in many funcions
has an accepted value. In particular, it should be a string that can only be:
'structural', 'functional' or 'joint'.
'''

def check_model_type(model_type):
    '''
    Check if the specified 'model_type' is valid.

    :param str model_type: String indicating the model type.

    :raises AssertionError: If 'model_type' is not one of 'structural', 'functional', or 'joint'.

    This function ensures that the provided 'model_type' is one of
    the allowed values: 'structural', 'functional', or 'joint'.
    If 'model_type' is not valid, an AssertionError is raised with a descriptive error message.
    '''

    assert model_type in ('structural','functional','joint'),(
        f"'{model_type}' is not a valid model_type: "
        "'model_type' must be either 'structural', 'functional' or 'joint'")
