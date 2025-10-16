import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os


def main():
    # Configuration de la page
    st.set_page_config(
        page_title="Détection de Visages - Viola-Jones",
        page_icon="📷",
        layout="wide"
    )

    # Titre principal
    st.title("🔍 Détection de Visages avec l'Algorithme de Viola-Jones")
    st.markdown("---")

    # Instructions pour l'utilisateur
    st.header("📋 Instructions d'utilisation")
    st.markdown("""
    1. **Téléchargez une image** contenant des visages en utilisant l'outil de téléchargement ci-dessous
    2. **Ajustez les paramètres** de détection selon vos besoins :
       - **Couleur des rectangles** : Choisissez la couleur des cadres autour des visages détectés
       - **Échelle (scaleFactor)** : Contrôle la réduction d'image à chaque recherche (1.01-1.5)
       - **Voisins minimum (minNeighbors)** : Détermine le nombre de détections voisines requises
    3. **Visualisez les résultats** : L'image avec les visages détectés s'affichera automatiquement
    4. **Téléchargez le résultat** : Enregistrez l'image annotée sur votre appareil

    **Conseils** :
    - Des valeurs plus élevées de `scaleFactor` accélèrent la détection mais peuvent manquer des visages
    - Des valeurs plus élevées de `minNeighbors` réduisent les faux positifs mais peuvent manquer des visages
    - Pour de meilleurs résultats, utilisez des images claires avec des visages bien visibles
    """)

    st.markdown("---")

    # Vérification de l'existence du classificateur
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        st.error(f"Fichier cascade non trouvé : {cascade_path}")
        st.stop()

    # Initialisation du classificateur en cascade
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Sidebar pour les paramètres
    st.sidebar.header("⚙️ Paramètres de Détection")

    # Sélecteur de couleur
    color = st.sidebar.color_picker("🎨 Couleur des rectangles", "#FF0000")
    # Conversion de l'hexadécimal en BGR pour OpenCV
    color_bgr = tuple(int(color.lstrip('#')[i:i + 2], 16) for i in (4, 2, 0))

    # Paramètres de détection
    scaleFactor = st.sidebar.slider(
        "📐 Échelle (scaleFactor)",
        min_value=1.01,
        max_value=1.5,
        value=1.1,
        step=0.01,
        help="Facteur de réduction d'image à chaque recherche. Des valeurs plus basses détectent plus de visages mais sont plus lentes."
    )

    minNeighbors = st.sidebar.slider(
        "👥 Voisins minimum (minNeighbors)",
        min_value=1,
        max_value=10,
        value=5,
        help="Nombre minimum de détections voisines requises. Des valeurs plus élevées réduisent les faux positifs."
    )

    # Téléchargement d'image
    st.header("📤 Téléchargement de l'Image")
    uploaded_file = st.file_uploader(
        "Choisissez une image",
        type=['jpg', 'jpeg', 'png'],
        help="Formats supportés : JPG, JPEG, PNG"
    )

    if uploaded_file is not None:
        # Conversion de l'image téléchargée en format OpenCV
        try:
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            # Conversion en niveaux de gris pour la détection
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Affichage de l'image originale
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🖼️ Image Originale")
                st.image(image, use_column_width=True)
                st.write(f"Dimensions : {img_array.shape[1]} x {img_array.shape[0]} pixels")

            # Détection des visages
            with st.spinner('🔍 Recherche des visages...'):
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scaleFactor,
                    minNeighbors=minNeighbors,
                    minSize=(30, 30)
                )

            # Dessiner les rectangles autour des visages détectés
            result_img = img_array.copy()
            for (x, y, w, h) in faces:
                # S'assurer que l'image a 3 canaux pour la couleur
                if len(result_img.shape) == 2:
                    result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2RGB)

                cv2.rectangle(result_img, (x, y), (x + w, y + h), color_bgr, 3)
                # Ajouter un numéro pour chaque visage détecté
                cv2.putText(result_img, f'Visage {len([f for f in faces if f[1] <= y])}',
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

            with col2:
                st.subheader("✅ Résultat de la Détection")
                st.image(result_img, use_column_width=True)
                st.write(f"**{len(faces)} visage(s) détecté(s)**")

                # Affichage des statistiques de détection
                if len(faces) > 0:
                    st.info(f"""
                    **Résultats de la détection :**
                    - Nombre total de visages détectés : **{len(faces)}**
                    - Couleur utilisée : {color}
                    - Paramètre scaleFactor : {scaleFactor}
                    - Paramètre minNeighbors : {minNeighbors}
                    """)
                else:
                    st.warning("""
                    **Aucun visage détecté.** Essayez de :
                    - Réduire la valeur de scaleFactor
                    - Réduire la valeur de minNeighbors
                    - Utiliser une image plus claire
                    """)

            # Fonction pour enregistrer l'image
            st.markdown("---")
            st.header("💾 Enregistrement du Résultat")

            if len(faces) > 0:
                # Conversion de l'image result_img en bytes pour le téléchargement
                if len(result_img.shape) == 3 and result_img.shape[2] == 3:
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    result_pil = Image.fromarray(result_img_rgb)
                else:
                    result_pil = Image.fromarray(result_img)

                buf = io.BytesIO()
                result_pil.save(buf, format='JPEG', quality=95)
                byte_im = buf.getvalue()

                # Bouton de téléchargement
                st.download_button(
                    label="📥 Télécharger l'image avec les détections",
                    data=byte_im,
                    file_name=f"visages_detectes_{len(faces)}_faces.jpg",
                    mime="image/jpeg",
                    help="Cliquez pour enregistrer l'image avec les visages détectés sur votre appareil"
                )

                st.success("L'image est prête à être téléchargée !")
            else:
                st.error("Aucun visage détecté - impossible de télécharger le résultat.")

        except Exception as e:
            st.error(f"Erreur lors du traitement de l'image : {str(e)}")

    else:
        # Message quand aucune image n'est téléchargée
        st.info("👆 Veuillez télécharger une image pour commencer la détection des visages.")

    # Pied de page
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            Application de détection de visages utilisant l'algorithme de Viola-Jones • 
            Développé avec Streamlit et OpenCV
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()