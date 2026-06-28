# Waste Type Identification

Progetto di classificazione automatica di immagini di rifiuti realizzato per l'esame di Machine Learning dell'Università degli Studi di Salerno (UNISA), A.A. 2025/2026.

L'obiettivo è addestrare e validare modelli in PyTorch capaci di riconoscere la classe di appartenenza di un oggetto di rifiuto a partire da immagini acquisite in condizioni diverse: risoluzioni differenti, sfondi variabili, immagini preprocessate, più oggetti della stessa classe e campioni non perfettamente rappresentativi del test set.

## Obiettivo del Progetto

Il sistema affronta il task "Waste type identification" indicato nella traccia del progetto. La soluzione deve classificare ogni immagine in una delle 8 classi previste, mantenendo un buon compromesso tra:

- balanced accuracy sul test set privato;
- memoria GPU richiesta in test;
- velocità di elaborazione dei campioni;
- semplicità e spiegabilità della pipeline adottata.

Il progetto non estende il training set fornito. Sono invece utilizzati split train/validation/test, cross-validation, data augmentation e fine-tuning di modelli pre-addestrati.

## Classi e Label

La corrispondenza tra label e classi segue la traccia e non deve essere modificata:

| Label | Classe |
| --- | --- |
| 0 | Battery |
| 1 | Clothing |
| 2 | Glass |
| 3 | Metal |
| 4 | Organic |
| 5 | Papery |
| 6 | Plastic |
| 7 | Undifferentiated |

Alcune classi sono multi-classe nella composizione del dataset:

- Clothing: clothes, shoes
- Papery: paper, cardboard
- Glass: brown, green, transparent

## Contenuto del Progetto

```
Progetto Machine Learning/
├── Waste_Classifier_Trainer.ipynb    # Notebook principale per training e validazione
├── config.yaml                       # Configurazione di default
├── README.md                         # Descrizione del progetto
├── requirements.txt                  # Dipendenze Python
├── scarica_dataset.py                # Script per scaricare il dataset da Google Drive
├── avvia_notebook.bat                # Avvio rapido su Windows (scarica dataset se mancante)
├── traccia.pdf                       # Traccia del progetto
├── waste_classifier/                 # Modulo Python riutilizzabile
│   ├── __init__.py
│   └── trainer.py                    # Dataset, modelli, training e salvataggio
├── experiments/                      # Risultati generati automaticamente
│   └── nome_modello_timestamp/
│       ├── config.yaml
│       ├── models/
│       ├── plots/
│       └── logs/
└── dataset/                          # Dataset estratto automaticamente dallo ZIP
```

> **Nota:** il file `waste_type_identification.zip` **non è incluso nel repository** per via delle dimensioni (≈ 240 MB). Segui la sezione **Setup** per scaricarlo.

La cartella `Vecchio progetto/` contiene materiale precedente e non fa parte della versione pulita del nuovo progetto.

## Funzionalità

Il notebook permette di configurare ed eseguire esperimenti completi:

- estrazione e analisi del dataset;
- split train/validation/test configurabile;
- K-Fold Cross Validation;
- data augmentation globale e specifica per classe;
- training in due fasi: feature extraction e fine-tuning;
- uso opzionale di Mixed Precision (AMP);
- early stopping;
- scheduler del learning rate;
- salvataggio di pesi, configurazione, metriche, grafici e report.
- salvataggio delle risorse usate dal modello durante training e inferenza.

## Modelli Supportati

- EfficientNet-B0
- EfficientNet-B2
- EfficientNet-B3
- MobileNetV3-Small

I modelli sono inizializzati con pesi pre-addestrati quando richiesto. Nella prima fase viene addestrato il classificatore finale; nella seconda fase possono essere sbloccati blocchi finali del backbone per il fine-tuning.

## Metrica di Valutazione

La metrica principale richiesta dalla traccia è la balanced accuracy, calcolata come media dei True Positive Rate delle 8 classi:

```text
Bal.Acc = (TPR_battery + TPR_clothing + TPR_glass + TPR_metal
           + TPR_organic + TPR_papery + TPR_plastic
           + TPR_undifferentiated) / 8
```

Questa metrica è adatta al problema perché riduce l'effetto di eventuali sbilanciamenti tra le classi.

## Requisiti

- Python 3.8+
- PyTorch 2.0+
- Jupyter o JupyterLab
- GPU NVIDIA consigliata

Installa le dipendenze con:

```bash
pip install -r requirements.txt
```

## Setup

### 1 — Clona il repository e installa le dipendenze

```bash
git clone <url-del-repo>
cd "Progetto Machine Learning"
pip install -r requirements.txt
```

### 2 — Scarica il dataset

Il dataset non è incluso nel repository (file troppo grande per GitHub).
È disponibile su Google Drive al link:

> <https://drive.google.com/file/d/1pu_Awz4QFIMHN86eN7UCxr1ZGJ4amzWD/view>

**Opzione A — script automatico (consigliato):**

```bash
python scarica_dataset.py
```

Lo script installa automaticamente `gdown` se necessario, scarica
`waste_type_identification.zip` e lo estrae nella cartella `dataset/`.

**Opzione B — download manuale:**

Scarica il file dal link sopra, salvalo nella root del progetto con il nome
`waste_type_identification.zip`, poi esegui di nuovo `scarica_dataset.py`
(si occuperà solo dell'estrazione).

### 3 — Avvia il notebook

```bash
jupyter lab Waste_Classifier_Trainer.ipynb
```

Su Windows puoi usare lo script che gestisce tutto in automatico
(download dataset incluso):

```bash
avvia_notebook.bat
```

## Workflow degli Esperimenti

1. Caricare il dataset o indicare il file ZIP fornito.
2. Scegliere lo split o abilitare la K-Fold Cross Validation.
3. Selezionare uno o più modelli da confrontare.
4. Configurare augmentation, batch size, learning rate, epoche e scheduler.
5. Eseguire feature extraction e fine-tuning.
6. Analizzare balanced accuracy, curve di training e matrice di confusione.
7. Salvare configurazione, pesi e risultati nella cartella `experiments/`.

## Output Generati

Per ogni esperimento vengono salvati:

- pesi del modello in formato `.pth`;
- configurazione effettivamente usata;
- cronologia del training in JSON;
- curve di loss e balanced accuracy;
- matrice di confusione;
- classification report per classe.
- report delle risorse in `logs/resource_usage.json`.

Il file `resource_usage.json` contiene, per ogni fase misurata, durata, batch size, epoche eseguite, RAM di processo se disponibile, memoria GPU allocata/riservata, picco di memoria GPU e throughput in immagini al secondo per l'inferenza.

## Vincoli della Traccia

La soluzione è pensata rispettando i vincoli principali indicati nel PDF:

- non estendere il training set fornito;
- usare una soluzione eseguibile in PyTorch;
- rendere spiegabili preprocessing, modello e strategia di training;
- mantenere il consumo di memoria compatibile con l'esecuzione in Google Colab;
- monitorare memoria e velocità per valutare il compromesso tra accuratezza, risorse e complessità computazionale;
- conservare il protocollo di split train/validation;
- produrre codice e pesi necessari alla valutazione finale.

## Note

Il progetto è organizzato per facilitare il confronto tra esperimenti. Ogni run viene salvata in una cartella separata con timestamp, così da mantenere traccia di configurazioni, metriche e risultati senza sovrascrivere prove precedenti.
