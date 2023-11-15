# adapted from https://github.com/lucabaldini/splrand/blob/master/setup.sh

#!/bin/bash

# See this stackoverflow question
# http://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within
# for the magic in this command
SETUP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#
# Base package root. All the other releavant folders are relative to this
# location.
#
export BRAIN_PREDICTION_ROOT=$SETUP_DIR
echo "BRAIN_PREDICTION_ROOT set to " $BRAIN_PREDICTION_ROOT

#
# Add the root folder to the $PYTHONPATH so that we can effectively import
# the relevant modules.
#
export PYTHONPATH=$BRAIN_PREDICTION_ROOT:$PYTHONPATH
echo "PYTHONPATH set to " $PYTHONPATH