#!/bin/bash

# Define the root directory as the current directory
ROOT_DIR=$(pwd)
OUTPUT_FILE="generated_project_structure.txt"

# List of files and patterns to be included
INCLUDE_FILES=(
   



   "*"



)

# List of patterns to exclude
EXCLUDE_PATTERNS=(
    "__pycache__"
    ".DS_Store"
  "_site"
    ".jekyll-cache"
    "OLD"
    "docs"

      "Gemfile.lock"
      ".git"
      "generate_project_structure.sh"
      "generated_project_structure.txt"
      ".gitignore‡"


)

# Initialize the output file
echo "$(basename "$ROOT_DIR")/" > $OUTPUT_FILE

# Function to check if a path should be excluded
should_exclude() {
    local FILE_PATH=$1
    for PATTERN in "${EXCLUDE_PATTERNS[@]}"; do
        if [[ "$FILE_PATH" == *"$PATTERN"* ]]; then
            return 0
        fi
    done
    return 1
}

# Function to check if a file should be included
should_include() {
    local FILE_PATH=$1
    local REL_PATH="${FILE_PATH#$ROOT_DIR/}"

    # First check if the path should be excluded
    if should_exclude "$REL_PATH"; then
        return 1
    fi

    # If include list is empty, include everything
    if [ ${#INCLUDE_FILES[@]} -eq 0 ]; then
        return 0
    fi

    # Check each include pattern
    for INCLUDE in "${INCLUDE_FILES[@]}"; do
        # Check for wildcard patterns
        if [[ "$INCLUDE" == *"*" ]]; then
            local PATTERN=${INCLUDE%"*"} # Remove the * from the end
            if [[ "$REL_PATH" == "$PATTERN"* ]]; then
                return 0
            fi
        # Check for exact file matches
        elif [ "$REL_PATH" == "$INCLUDE" ]; then
            return 0
        fi
    done

    # Special handling for directories that need to be shown
    # to maintain the tree structure for included files
    if [ -d "$FILE_PATH" ]; then
        for INCLUDE in "${INCLUDE_FILES[@]}"; do
            if [[ "$INCLUDE" == "$REL_PATH/"* ]]; then
                return 0
            fi
        done
    fi

    return 1
}

# Function to recursively print directory structure
print_structure() {
    local DIR=$1
    local PREFIX=$2
    local FILE

    # List directories and files, including hidden ones
    shopt -s dotglob
    for FILE in "$DIR"/*; do
        shopt -u dotglob
        if ! should_include "$FILE"; then
            continue
        fi
        if [ -d "$FILE" ]; then
            echo "${PREFIX}├── $(basename "$FILE")/" >> $OUTPUT_FILE
            print_structure "$FILE" "${PREFIX}│   "
        else
            echo "${PREFIX}├── $(basename "$FILE")" >> $OUTPUT_FILE
        fi
    done
}

# Function to print file paths and contents
print_file_content() {
    local DIR=$1
    local FILE

    # List files including hidden ones
    shopt -s dotglob
    for FILE in "$DIR"/*; do
        shopt -u dotglob
        if ! should_include "$FILE"; then
            continue
        fi
        if [ -d "$FILE" ]; then
            print_file_content "$FILE"
        else
            REL_PATH="${FILE#$ROOT_DIR/}"
            echo -e "\n// ${REL_PATH}:\n$(cat "$FILE")" >> $OUTPUT_FILE
        fi
    done
}

# Generate the directory structure
print_structure "$ROOT_DIR" "│   "

# Add file paths and contents
print_file_content "$ROOT_DIR"

echo "Project structure and contents have been saved to $OUTPUT_FILE"

# Copy the content to the clipboard
pbcopy < $OUTPUT_FILE
echo "The project structure and contents have been copied to the clipboard."