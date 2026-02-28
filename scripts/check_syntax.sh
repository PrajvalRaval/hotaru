#!/bin/bash
# Hotaru Syntax Checker
# This script ensures that all Python files in the project have no syntax or indentation errors.

echo "🔍 Checking Hotaru codebase for syntax/indentation errors..."
FAILED=0

# Find all python files excluding venv
PY_FILES=$(find . -name "*.py" -not -path "./venv/*")

for file in $PY_FILES; do
    if python3 -m py_compile "$file" > /dev/null 2>&1; then
        echo "✅ $file: OK"
    else
        echo "❌ $file: SYNTAX/INDENTATION ERROR DETECTED!"
        python3 -m py_compile "$file" # Show the error
        FAILED=1
    fi
done

if [ $FAILED -eq 1 ]; then
    echo "🚨 Some files failed the syntax check. Please fix them."
    exit 1
else
    echo "✨ All files passed the syntax check."
    exit 0
fi
