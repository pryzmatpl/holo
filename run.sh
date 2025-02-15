#!/bin/bash
# run.sh: Set LD_LIBRARY_PATH and run a chosen executable with additional arguments.

# Ensure JAVA_HOME is set; if not, set it here:
# export JAVA_HOME="/usr/lib/jvm/java-23-openjdk"

# Set the required library path.
export LD_PRELOAD=/usr/lib/libpython3.12.so
export LD_LIBRARY_PATH="$JAVA_HOME/lib:$JAVA_HOME/lib/server:/usr/local/lib:/usr/lib:$LD_LIBRARY_PATH"
echo "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"

# Check that at least one argument (target) is provided.
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 {holo|holorunner|tests} [arguments...]"
    exit 1
fi

# The first argument is the target.
target=$1
shift  # Remove the target from the argument list; remaining args will be passed along.

case $target in
    holo)
        if [ -x "./build/holo" ]; then
            echo "Running holo with arguments: $@"
            ./build/holo "$@"
        else
            echo "Executable './holo' not found. Please build the project first."
        fi
        ;;
    holorunner)
        if [ -x "./build/holorunner" ]; then
            echo "Running holorunner with arguments: $@"
            ./build/holorunner "$@"
        else
            echo "Executable './holorunner' not found. Please build the project first."
        fi
        ;;
    tests)
        if [ -x "./build/runTests" ]; then
            echo "Running tests with arguments: $@"
            ./build/runTests "$@"
        else
            echo "Executable './runTests' not found. Please build the project first."
        fi
        ;;
    *)
        echo "Invalid target: '$target'"
        echo "Usage: $0 {holo|holorunner|tests} [arguments...]"
        exit 1
        ;;
esac
