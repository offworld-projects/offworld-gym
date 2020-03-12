#!/bin/bash

# (Creates python callable GRPC logic from the .proto file)
# You can run this script from anywhere, and it will generate the new files in the same folder as this script.


THIS_SCRIPTS_DIR="`dirname \"$0\"`"

# Get the sitepackages path where the numproto is imported from
NUMPROTO_PACKAGES_PATH=$(python3 -c 'import os, numproto; print(os.path.abspath(f"{os.path.dirname(numproto.__file__)}/.."))')

python3 -m grpc_tools.protoc \
  -I "$NUMPROTO_PACKAGES_PATH" \
  -I "$THIS_SCRIPTS_DIR" \
  --python_out="$THIS_SCRIPTS_DIR" \
  --grpc_python_out="$THIS_SCRIPTS_DIR" \
  remote_env.proto
