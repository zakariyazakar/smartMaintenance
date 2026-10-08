from pathlib import Path

import pandas as pd
import streamlit as st

from src.vibration import inspect_window
from src.vision import inspect_image
from src.history import save_alert


# =========================================================
# 1. CONFIGURATION GENERALE
# =========================================================

st.set_page_config(
    page_title="Smart Maintenance",
    layout="wide"
)


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR.parent / "Data"

HISTORY_FILE = (
    BASE_DIR
    / "data"
    / "alerts_history.csv"
)


# =========================================================
# 2. TITRE
# =========================================================

st.title("Smart Maintenance")

st.write(
    "Système intelligent d'inspection "
    "et de maintenance prédictive"
)


# =========================================================
# 3. CONFIGURATION DES ZONES
# =========================================================

zones = {

    "Zone A": {
        "generator": "G1",
        "cameras": [
            "CAM-A1",
            "CAM-A2",
            "CAM-A3"
        ]
    },

    "Zone B": {
        "generator": "G2",
        "cameras": [
            "CAM-B1",
            "CAM-B2",
            "CAM-B3"
        ]
    },

    "Zone C": {
        "generator": "G3",
        "cameras": [
            "CAM-C1",
            "CAM-C2",
            "CAM-C3"
        ]
    }
}


# =========================================================
# 4. FICHIERS VIBRATION
# =========================================================

generator_files = {

    "G1":
        DATA_DIR / "NormalBaseline0.mat",

    "G2":
        DATA_DIR / "Fault0.007InnerBall.mat",

    "G3":
        DATA_DIR / "Fault0.007InnerRace.mat"
}


# =========================================================
# 5. IMAGES DES CAMERAS
# =========================================================

camera_images = {

    "CAM-A1":
        DATA_DIR / "cameras" / "cam_a1.jpg",

    "CAM-A2":
        DATA_DIR / "cameras" / "cam_a2.jpg",

    "CAM-A3":
        DATA_DIR / "cameras" / "cam_a3.jpg",

    "CAM-B1":
        DATA_DIR / "cameras" / "cam_b1.jpg",

    "CAM-B2":
        DATA_DIR / "cameras" / "cam_b2.jpg",

    "CAM-B3":
        DATA_DIR / "cameras" / "cam_b3.jpg",

    "CAM-C1":
        DATA_DIR / "cameras" / "cam_c1.jpg",

    "CAM-C2":
        DATA_DIR / "cameras" / "cam_c2.jpg",

    "CAM-C3":
        DATA_DIR / "cameras" / "cam_c3.jpg"
}


# =========================================================
# 6. SESSION STATE
# =========================================================

if "generator_window_index" not in st.session_state:

    st.session_state.generator_window_index = {
        "G1": 0,
        "G2": 0,
        "G3": 0
    }


if "generator_results" not in st.session_state:

    st.session_state.generator_results = {}


if "previous_generator_state" not in st.session_state:

    st.session_state.previous_generator_state = {
        "G1": None,
        "G2": None,
        "G3": None
    }


# =========================================================
# 7. SURVEILLANCE DES GENERATEURS
# =========================================================

st.header(
    "Surveillance des générateurs"
)


@st.fragment(run_every="5s")
def monitor_generators():

    cols = st.columns(3)

    for i, generator_name in enumerate(
        ["G1", "G2", "G3"]
    ):

        generator_file = generator_files[
            generator_name
        ]


        # ---------------------------------------------
        # Vérifier que le fichier existe
        # ---------------------------------------------

        if not generator_file.exists():

            with cols[i]:

                st.subheader(
                    generator_name
                )

                st.error(
                    "Fichier vibration introuvable"
                )

            continue


        # ---------------------------------------------
        # Fenêtre actuelle
        # ---------------------------------------------

        window_index = (
            st.session_state
            .generator_window_index[
                generator_name
            ]
        )


        # ---------------------------------------------
        # Analyser une fenêtre
        # ---------------------------------------------

        try:

            result = inspect_window(
                generator_file,
                window_index
            )

        except Exception as error:

            with cols[i]:

                st.subheader(
                    generator_name
                )

                st.error(
                    f"Erreur : {error}"
                )

            continue


        # ---------------------------------------------
        # Sauvegarder le dernier résultat
        # ---------------------------------------------

        st.session_state.generator_results[
            generator_name
        ] = result


        diagnosis = result[
            "prediction"
        ]


        # =============================================
        # HISTORIQUE DES ALERTES
        # =============================================

        previous_state = (
            st.session_state
            .previous_generator_state[
                generator_name
            ]
        )


        # Sauvegarder seulement une nouvelle anomalie
        if (
            diagnosis != "Normal"
            and diagnosis != previous_state
        ):

            save_alert(
                generator=generator_name,
                fault=diagnosis,
                window_index=(
                    result[
                        "window_index"
                    ] + 1
                )
            )


        # Mettre à jour l'état précédent
        st.session_state.previous_generator_state[
            generator_name
        ] = diagnosis


        # =============================================
        # AFFICHAGE
        # =============================================

        with cols[i]:

            st.subheader(
                generator_name
            )


            if diagnosis == "Normal":

                st.success(
                    "Fonctionnement normal"
                )

            else:

                st.error(
                    f"Anomalie : {diagnosis}"
                )


            st.write(
                "Fenêtre analysée :",
                f"{result['window_index'] + 1}"
                f" / {result['n_windows']}"
            )


            if "probabilities" in result:

                best_probability = max(
                    result[
                        "probabilities"
                    ].values()
                )

                st.write(
                    "Score modèle :",
                    f"{best_probability * 100:.1f}%"
                )


        # ---------------------------------------------
        # Fenêtre suivante
        # ---------------------------------------------

        st.session_state.generator_window_index[
            generator_name
        ] += 1


monitor_generators()


# =========================================================
# 8. INSPECTION D'UNE ZONE
# =========================================================

st.divider()

st.header(
    "Inspection d'une zone"
)


selected_zone = st.selectbox(
    "Choisir une zone",
    list(zones.keys())
)


zone_info = zones[
    selected_zone
]


generator_name = zone_info[
    "generator"
]


camera_names = zone_info[
    "cameras"
]


# =========================================================
# 9. INFORMATIONS DE LA ZONE
# =========================================================

st.subheader(
    selected_zone
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Générateur associé",
        generator_name
    )


with col2:

    st.metric(
        "Nombre de caméras",
        len(camera_names)
    )


generator_result = (
    st.session_state
    .generator_results
    .get(generator_name)
)


with col3:

    if generator_result is not None:

        st.metric(
            "État vibration",
            generator_result[
                "prediction"
            ]
        )

    else:

        st.metric(
            "État vibration",
            "En attente..."
        )


# =========================================================
# 10. ETAT DU GENERATEUR DE LA ZONE
# =========================================================

if generator_result is not None:

    diagnosis = generator_result[
        "prediction"
    ]

    if diagnosis == "Normal":

        st.success(
            f"{generator_name} : fonctionnement normal"
        )

    else:

        st.error(
            f"{generator_name} : anomalie vibration "
            f"({diagnosis})"
        )

    st.caption(
        "Dernière fenêtre analysée : "
        f"{generator_result['window_index'] + 1}"
        f" / {generator_result['n_windows']}"
    )

else:

    st.warning(
        "Aucune donnée vibration disponible."
    )


# =========================================================
# 11. CAMERAS
# =========================================================

st.divider()

st.header(
    f"Caméras - {selected_zone}"
)


camera_cols = st.columns(
    len(camera_names)
)


for i, camera_name in enumerate(
    camera_names
):

    image_path = camera_images.get(
        camera_name
    )

    with camera_cols[i]:

        st.subheader(
            camera_name
        )

        if image_path is None:

            st.error(
                "Caméra non configurée"
            )

            continue

        if image_path.exists():

            st.image(
                str(image_path),
                caption=camera_name,
                use_container_width=True
            )

        else:

            st.error(
                "Image introuvable"
            )


# =========================================================
# 12. INSPECTION VISUELLE
# =========================================================

st.divider()

st.header(
    "Résultats de l'inspection visuelle"
)


for camera_name in camera_names:

    image_path = camera_images.get(
        camera_name
    )


    if image_path is None:

        continue


    if not image_path.exists():

        continue


    st.subheader(
        f"Inspection - {camera_name}"
    )


    try:

        vision_results = inspect_image(
            str(image_path)
        )

    except Exception as error:

        st.error(
            f"Erreur vision : {error}"
        )

        continue


    if len(vision_results) == 0:

        st.warning(
            "Aucun équipement détecté."
        )

        continue


    # =====================================================
    # RESULTATS DES EQUIPEMENTS
    # =====================================================

    for item in vision_results:

        equipment_col, detection_col, fault_col = (
            st.columns(
                [2, 1, 1]
            )
        )


        with equipment_col:

            st.write(
                f"**{item['equipment']}**"
            )


        with detection_col:

            st.write(
                "Détection : "
                f"{item['detection_confidence'] * 100:.1f}%"
            )


        with fault_col:

            fault = item[
                "fault"
            ]


            if fault is None:

                st.info(
                    "Non analysé"
                )


            elif fault in [
                "good",
                "normal"
            ]:

                st.success(
                    fault
                )


            else:

                st.error(
                    fault
                )


        if item[
            "fault_confidence"
        ] is not None:

            st.caption(
                "Score classification : "
                f"{item['fault_confidence'] * 100:.2f}%"
            )


        st.divider()


# =========================================================
# 13. HISTORIQUE DES ALERTES
# =========================================================

st.header(
    "Historique des alertes"
)


if HISTORY_FILE.exists():

    try:

        history_df = pd.read_csv(
            HISTORY_FILE
        )


        if len(history_df) > 0:

            # Les plus récentes en premier
            history_df = (
                history_df
                .iloc[::-1]
                .reset_index(drop=True)
            )


            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Aucune alerte enregistrée."
            )

    except Exception as error:

        st.error(
            f"Impossible de lire l'historique : {error}"
        )

else:

    st.info(
        "Aucune alerte enregistrée."
    )