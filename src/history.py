from pathlib import Path
from datetime import datetime
import csv


BASE_DIR = Path(__file__).resolve().parent.parent

HISTORY_DIR = BASE_DIR / "data"

HISTORY_FILE = HISTORY_DIR / "alerts_history.csv"


def save_alert(
    generator,
    fault,
    window_index
):

    # Créer le dossier s'il n'existe pas
    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Vérifier si le fichier existe déjà
    file_exists = HISTORY_FILE.exists()


    # Ouvrir le CSV en mode ajout
    with open(
        HISTORY_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)


        # Ajouter l'en-tête seulement
        # lors de la première création
        if not file_exists:

            writer.writerow([
                "timestamp",
                "generator",
                "fault",
                "window_index"
            ])


        # Ajouter l'alerte
        writer.writerow([
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            generator,
            fault,
            window_index
        ])