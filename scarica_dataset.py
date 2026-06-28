"""
scarica_dataset.py
------------------
Scarica il dataset `waste_type_identification.zip` da Google Drive
ed estraelo nella cartella `dataset/`.

Uso:
    python scarica_dataset.py

Dipendenze:
    pip install gdown
"""

import os
import sys
import zipfile

# ── Configurazione ────────────────────────────────────────────────────────────
GDRIVE_FILE_ID = "1pu_Awz4QFIMHN86eN7UCxr1ZGJ4amzWD"
ZIP_FILENAME   = "waste_type_identification.zip"
DATASET_DIR    = "dataset"
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH   = os.path.join(SCRIPT_DIR, ZIP_FILENAME)
DEST_DIR   = os.path.join(SCRIPT_DIR, DATASET_DIR)


def _try_gdown(file_id: str, dest: str) -> bool:
    """Scarica con gdown (gestisce i cookie di Google Drive)."""
    try:
        import gdown
    except ImportError:
        print("[INFO] gdown non trovato. Installo...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[INFO] Download da Google Drive  →  {dest}")
    gdown.download(url, dest, quiet=False, fuzzy=True)
    return os.path.isfile(dest) and os.path.getsize(dest) > 0


def download_zip() -> None:
    """Scarica il file ZIP se non è già presente."""
    if os.path.isfile(ZIP_PATH):
        print(f"[OK] ZIP già presente: {ZIP_PATH}")
        return

    print("=" * 60)
    print("  Download dataset waste_type_identification.zip")
    print("=" * 60)

    success = _try_gdown(GDRIVE_FILE_ID, ZIP_PATH)

    if not success:
        print("[ERRORE] Download fallito. Scarica manualmente il file da:")
        print("  https://drive.google.com/file/d/1pu_Awz4QFIMHN86eN7UCxr1ZGJ4amzWD/view")
        print(f"  e salvalo come: {ZIP_PATH}")
        sys.exit(1)

    print(f"[OK] Download completato: {ZIP_PATH}")


def extract_zip() -> None:
    """Estrae il file ZIP nella cartella dataset/."""
    if os.path.isdir(DEST_DIR) and os.listdir(DEST_DIR):
        print(f"[OK] Dataset già estratto in: {DEST_DIR}")
        return

    print(f"[INFO] Estrazione  →  {DEST_DIR}")
    os.makedirs(DEST_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DEST_DIR)
    print(f"[OK] Estrazione completata: {DEST_DIR}")


if __name__ == "__main__":
    download_zip()
    extract_zip()
    print("\n✅ Dataset pronto all'uso.")
