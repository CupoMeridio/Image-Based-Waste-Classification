import json
import io

with open('Waste_Classifier_Trainer.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "# Seleziona automaticamente GPU se disponibile, altrimenti CPU.\n)" in source:
            source = source.replace("# Seleziona automaticamente GPU se disponibile, altrimenti CPU.\n)", ")\n\n# Seleziona automaticamente GPU se disponibile, altrimenti CPU.")
            cell['source'] = [line for line in io.StringIO(source)]
            print("Commento spostato con successo.")

with open('Waste_Classifier_Trainer.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
