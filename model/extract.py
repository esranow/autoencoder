import json
import sys

nb_path = sys.argv[1]
out_path = sys.argv[2]
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

extracted = ""
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        if 'class ' in source or 'import ' in source or 'def ' in source:
            extracted += source + '\n' + '-'*40 + '\n'

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(extracted)
