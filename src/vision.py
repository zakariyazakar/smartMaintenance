from pathlib import Path
from ultralytics import YOLO


# =========================================================
# 1. RACINE DU PROJET
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# =========================================================
# 2. CHEMINS DES MODELES
# =========================================================

DETECTOR_PATH = (
    BASE_DIR
    / "models"
    / "vision"
    / "detector"
    / "detector.pt"
)

CLASSIFIERS_DIR = (
    BASE_DIR
    / "models"
    / "vision"
    / "classifiers"
)


# =========================================================
# 3. CHARGEMENT DU DETECTEUR
# =========================================================

detector = YOLO(DETECTOR_PATH)


# =========================================================
# 4. CHARGEMENT DES CLASSIFICATEURS
# =========================================================

classifiers = {
    "vari-grip": YOLO(
        CLASSIFIERS_DIR / "vari_grip.pt"
    ),

    "glass insulator": YOLO(
        CLASSIFIERS_DIR / "glass_insulator.pt"
    ),

    "lightning rod suspension": YOLO(
        CLASSIFIERS_DIR / "lightning_rod_suspension.pt"
    ),

    "polymer insulator upper shackle": YOLO(
        CLASSIFIERS_DIR / "polymer_upper_shackle.pt"
    ),

    "yoke suspension": YOLO(
        CLASSIFIERS_DIR / "yoke_suspension.pt"
    ),
}
def detect_equipments(image_path, confidence=0.4):

    # Le détecteur analyse l'image
    results = detector.predict(
        source=image_path,
        conf=confidence,
        imgsz=640,
        device="cpu",
        verbose=False
    )

    # Une image => on récupère son résultat
    result = results[0]

    detections = []

    # On parcourt toutes les bounding boxes trouvées
    for box in result.boxes:

        # Numéro de la classe détectée
        class_id = int(box.cls[0])

        # Exemple : 7 -> "glass insulator"
        equipment_name = result.names[class_id]

        # Confiance de la détection
        detection_confidence = float(box.conf[0])

        # Coordonnées de la bounding box
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].cpu().numpy()
        )

        detection = {
            "equipment": equipment_name,
            "confidence": detection_confidence,
            "bbox": (x1, y1, x2, y2)
        }

        detections.append(detection)

    return detections

def inspect_image(image_path, detection_confidence=0.4):

    import cv2

    # Charger l'image avec OpenCV
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Impossible de charger l'image : {image_path}"
        )

    # Faire les détections
    detections = detect_equipments(
        image_path,
        confidence=detection_confidence
    )

    inspections = []

    # Parcourir chaque équipement détecté
    for detection in detections:

        equipment = detection["equipment"]

        x1, y1, x2, y2 = detection["bbox"]

        # Découper seulement l'équipement
        crop = image[
            y1:y2,
            x1:x2
        ]

        inspection = {
            "equipment": equipment,
            "detection_confidence": detection["confidence"],
            "bbox": detection["bbox"],
            "fault": None,
            "fault_confidence": None
        }

        # Vérifier si un classifier existe
        if equipment in classifiers:

            classifier = classifiers[equipment]

            # Classifier le crop
            results = classifier.predict(
                source=crop,
                imgsz=224,
                device="cpu",
                verbose=False
            )

            result = results[0]

            # Classe ayant la probabilité la plus élevée
            class_id = int(result.probs.top1)

            fault_name = result.names[class_id]

            fault_confidence = float(
                result.probs.top1conf
            )

            inspection["fault"] = fault_name

            inspection["fault_confidence"] = fault_confidence

        inspections.append(inspection)

    return inspections

# =========================================================
# 5. TEST
# =========================================================

if __name__ == "__main__":

    image_path = "238-2_DJI_0317.jpg"

    inspections = inspect_image(image_path)

    print("\n==============================")
    print("INSPECTION COMPLETE")
    print("==============================")

    for item in inspections:

        print("\nEquipement :", item["equipment"])

        print(
            "Confiance détection :",
            f'{item["detection_confidence"]:.2%}'
        )

        if item["fault"] is not None:

            print(
                "Etat :",
                item["fault"]
            )

            print(
                "Confiance classification :",
                f'{item["fault_confidence"]:.2%}'
            )

        else:

            print(
                "Analyse défaut : non disponible"
            )