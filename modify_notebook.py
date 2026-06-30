import json
import io

with open('Waste_Classifier_Trainer.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Modifica 1: Aggiungere use_focal_loss e focal_gamma_widget
        if "n_splits = widgets.IntSlider(" in source and "use_focal_loss =" not in source:
            target = "n_splits = widgets.IntSlider(\n    value=default_config.get('kfold', {}).get('n_splits', 3), min=2, max=10, step=1,\n    description='N folds:',\n    layout=widgets.Layout(width='auto'),\n    style={'description_width': 'initial'}\n)\n"
            replacement = target + "\nuse_focal_loss = widgets.Checkbox(\n    value=default_config.get('training', {}).get('focal_loss', {}).get('enabled', False),\n    description='Usa Focal Loss',\n    style={'description_width': 'initial'}\n)\n\nfocal_gamma_widget = widgets.FloatSlider(\n    value=default_config.get('training', {}).get('focal_loss', {}).get('gamma', 2.0), min=0.0, max=5.0, step=0.5,\n    description='Focal gamma (γ):',\n    layout=widgets.Layout(width='auto'),\n    style={'description_width': 'initial'}\n)\n"
            source = source.replace(target, replacement)
        
        # Modifica 2: update_training_config
        if "def update_training_config(change=None):" in source:
            target = "    kfold_mode.layout.display = 'flex' if show_kfold else 'none'\n"
            replacement = target + "    focal_gamma_widget.layout.display = 'flex' if use_focal_loss.value else 'none'\n"
            source = source.replace(target, replacement)
            
            target2 = "use_kfold.observe(update_training_config)\n"
            replacement2 = target2 + "use_focal_loss.observe(update_training_config)\n"
            source = source.replace(target2, replacement2)
            
        # Modifica 3: display(widgets.VBox)
        if "widgets.HTML('<h4>Scheduler</h4>'),\n" in source and "Funzione di Loss" not in source:
            target = "    widgets.HTML('<h4>Scheduler</h4>'),\n"
            replacement = "    widgets.HTML('<h4>Funzione di Loss</h4>'),\n    use_focal_loss,\n    focal_gamma_widget,\n" + target
            source = source.replace(target, replacement)
            
        # Modifica 4: build_criterion
        if "fold_criterion = nn.CrossEntropyLoss(weight=fold_weights)" in source:
            target = "fold_criterion = nn.CrossEntropyLoss(weight=fold_weights)"
            replacement = "fold_criterion = build_criterion(\n                            use_focal=use_focal_loss.value,\n                            class_weights=fold_weights,\n                            gamma=focal_gamma_widget.value,\n                            label_smoothing=config['training'].get('label_smoothing', 0.0)\n                        )"
            source = source.replace(target, replacement)
            
        if "final_criterion = nn.CrossEntropyLoss(weight=final_weights)" in source:
            target = "final_criterion = nn.CrossEntropyLoss(weight=final_weights)"
            replacement = "final_criterion = build_criterion(\n                            use_focal=use_focal_loss.value,\n                            class_weights=final_weights,\n                            gamma=focal_gamma_widget.value,\n                            label_smoothing=config['training'].get('label_smoothing', 0.0)\n                        )"
            source = source.replace(target, replacement)
            
        if "criterion = nn.CrossEntropyLoss(weight=train_weights)" in source:
            target = "criterion = nn.CrossEntropyLoss(weight=train_weights)"
            replacement = "criterion = build_criterion(\n                        use_focal=use_focal_loss.value,\n                        class_weights=train_weights,\n                        gamma=focal_gamma_widget.value,\n                        label_smoothing=config['training'].get('label_smoothing', 0.0)\n                    )"
            source = source.replace(target, replacement)
            
        # Modifica 5: imports
        if "from waste_classifier import (" in source and "build_criterion" not in source:
            target = "from waste_classifier import ("
            replacement = "from waste_classifier import (\n    FocalLoss,\n    build_criterion,"
            source = source.replace(target, replacement)
            
        lines = [line for line in io.StringIO(source)]
        cell['source'] = lines

with open('Waste_Classifier_Trainer.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook modificato con successo.")
