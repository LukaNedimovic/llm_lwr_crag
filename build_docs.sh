#!/bin/bash

# This script is used to build HTML documentation for source code.
# It will automatically open the documentation using Firefox web browser.

# Make
cd $DOCS_DIR/
make clean

echo "Building documentation..."
make html SPHINXOPTS="-q" >/dev/null 2>&1 && echo "Sucessfully built the documentation!"

# Show the documentation
echo "Opening documentation!"
cd $DOCS_DIR/build/
firefox index.html
