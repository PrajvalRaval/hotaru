#!/bin/bash
# Hotaru Syntax Checker
# This script ensures that the main Python files have no syntax or indentation errors.

FILES=("app.py" "transcribe_engine.py" "convert_anime_whisper.py")

echo "🔍 Checking Hotaru codebase for syntax/indentation errors..."
FAILED=0

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        if python3 -m py_compile "$file" > /dev/null 2>&1; then
            echo "✅ $file: OK"
        else
            echo "❌ $file: SYNTAX/INDENTATION ERROR DETECTED!"
            python3 -m py_compile "$file" # Show the error
            FAILED=1
        fi
    fi
done

if [ $FAILED -eq 1 ]; then
    echo "🚨 Some files failed the syntax check. Please fix them."
    exit 1
else
    echo "✨ All files passed the syntax check."
    exit 0
fi
