import json
import io

with open('Waste_Classifier_Trainer.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Pattern esatto (ogni riga separata nel file sorgente della cella)
old_input = (
    "                if 'b0' in model_name:\n"
    "                    input_size = 224\n"
    "                elif 'b2' in model_name:\n"
    "                    input_size = 288\n"
    "                elif 'b3' in model_name:\n"
    "                    input_size = 300\n"
    "                else:\n"
    "                    input_size = 224"
)

new_input = (
    "                # input_size standard per architettura (valori fissi delle specifiche ufficiali)\n"
    "                _input_sizes = {'efficientnet_b0': 224, 'efficientnet_b2': 288, 'efficientnet_b3': 300, 'mobilenet_v3_small': 224}\n"
    "                input_size = _input_sizes.get(model_name, 224)"
)

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if old_input in source:
            source = source.replace(old_input, new_input)
            found = True
            cell['source'] = [line for line in io.StringIO(source)]
            print("Fix input_size applicato correttamente.")
            break

if not found:
    print("Pattern non trovato.")

with open('Waste_Classifier_Trainer.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
