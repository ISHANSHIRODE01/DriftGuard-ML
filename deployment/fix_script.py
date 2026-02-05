import re

# Read the file
with open('preprocessing_pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the problematic line
content = content.replace(
    "        pct = summary['split_percentages'][split]",
    "        pct = summary['split_percentages'].get(split, '')"
)

# Write back
with open('preprocessing_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed!")
