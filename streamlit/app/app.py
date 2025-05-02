import os
import glob
from pathlib import Path
import streamlit as st
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models

# comando para lanzar la app
#  streamlit run .\streamlit\app\app.py


# --- Configuraciones de la app ---
num_classes = 4
Images_size = 224
Images_types = ['jpg', 'jpeg', 'png']
ROOT_DIR = Path(__file__).resolve().parents[2]  # PracticaImagen/
models_dir = ROOT_DIR / 'models'

classnames = [
    "cataract", "diabetic_retinopathy", "glaucoma", "normal"
]

class_emojis = {
    "cataract": "🛏️", "diabetic_retinopathy": "🏖️", "glaucoma": "🌳", "normal": "🛣️"
}

# --- Funciones auxiliares ---
def listar_modelos(models_dir):
    pt_files = glob.glob(os.path.join(models_dir, "*.pt"))
    model_names = [os.path.splitext(os.path.basename(f))[0] for f in pt_files]
    return model_names

# --- Definición de la arquitectura de la CNN ---
class CNN(nn.Module):
    def __init__(self, base_model, num_classes, unfreezed_layers=0):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        # Congelar los parámetros del modelo base
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Descongelar las últimas capas si se requiere
        if unfreezed_layers > 0:
            for layer in list(self.base_model.children())[-unfreezed_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # Nueva capa fully connected personalizada
        self.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )

        # Reemplazar la capa fc original por una identidad
        self.base_model.fc = nn.Identity()

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --- Dataset personalizado para la imagen cargada ---
class CustomImageDataset(Dataset):
    def __init__(self, image, transform=None):
        self.image = image
        self.transform = transform

    def __len__(self):
        return 1  # Solo una imagen

    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(self.image)
        else:
            image = self.image
        # Etiqueta dummy (no se utiliza en la inferencia)
        label = 0
        return image, label

# --- Función principal ---
def main():
    st.set_page_config(page_title="Clasificador de Imágenes de Enfermedades Oculares", layout="centered", page_icon="📸")

    # Header con estilo
    st.markdown("""
    <style>
    .header {
        font-size: 40px;
        color: #2E86C1;
        padding: 20px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="header">📸 Clasificador de Imágenes de Enfermedades Oculares 👁️</h1>', unsafe_allow_html=True)

    # Tarjeta de introducción
    with st.container():
        st.markdown("""
        Sube una imagen y nuestro sistema inteligente te dirá a qué enfermedad ocular pertenece.
        """)
        st.markdown("---")

    # Barra lateral mejorada
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")
        st.markdown("Selecciona el modelo de IA que prefieras:")

        modelos_disponibles = listar_modelos(models_dir)
        if not modelos_disponibles:
            st.error("❌ No se encontraron modelos en la carpeta: " + models_dir)
            return

        selected_model_name = st.selectbox(
            "Modelo AI:",
            modelos_disponibles,
            help="Elige entre nuestros modelos entrenados para diferentes escenarios"
        )

        st.markdown("---")
        st.markdown("### 🔍 Categorías disponibles")
        with st.expander("Ver todas las categorías"):
            for idx, category in enumerate(classnames):
                st.markdown(f"{class_emojis[category]} **{category}**")

    # Sección principal de carga de imagen
    st.markdown("## 📤 Sube tu imagen")
    image_file = st.file_uploader("Arrastra o selecciona una imagen...", type=Images_types, 
                                help="Formatos soportados: JPG, JPEG, PNG")

    if image_file is not None:
        with st.spinner("🔍 Analizando imagen..."):
            image = Image.open(image_file).convert("RGB")
            transform_pipeline = transforms.Compose([
                transforms.Resize((Images_size, Images_size)),
                transforms.ToTensor()
            ])
            dataset = CustomImageDataset(image, transform=transform_pipeline)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

        with st.spinner("🧠 Cargando modelo IA..."):
            model_path = os.path.join(models_dir, selected_model_name + ".pt")
            if not os.path.exists(model_path):
                st.error(f"⚠️ Error: No se encontró el modelo en {model_path}")
                return

            # Detección automática de la arquitectura basada en el nombre del modelo
            model_name_lower = selected_model_name.lower()
            if "resnext101_64x4d" in model_name_lower:
                base_model = models.resnext101_64x4d(pretrained=False)
            elif "resnet50" in model_name_lower:
                base_model = models.resnet50(pretrained=False)
            elif "regnet_y_128gf" in model_name_lower:
                base_model = models.regnet_y_128gf(pretrained=False)
            elif "resnext50_32x4d" in model_name_lower:
                base_model = models.resnext50_32x4d(pretrained=False)
            else:
                st.warning("Modelo no reconocido, usando resnext101_64x4d por defecto")
                base_model = models.resnext101_64x4d(pretrained=False)

            model = CNN(base_model, num_classes)
            model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
            model.eval()


            with torch.no_grad():
                for img, _ in loader:
                    outputs = model(img)
                    _, top_class = torch.max(outputs, dim=1)
                    predicted_label = top_class.item()
                    class_name = classnames[predicted_label]
                    prob = outputs[0][predicted_label].item()

        # Mostrar resultados con estilo
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption='Tu imagen', use_column_width=True)

        with col2:
            emoji = class_emojis.get(class_name, "❓")
            st.markdown(f"## Resultado de Análisis:")
            st.markdown(f"<h2 style='color:#2E86C1;'>{emoji} {class_name}</h2>", unsafe_allow_html=True)
            st.metric(label="Confianza del modelo", value=f"{prob:.2%}")
            
            if prob > 0.75:
                st.success("¡Clasificación de alta confianza!")
            elif prob > 0.5:
                st.warning("Clasificación moderada")
            else:
                st.error("Baja confianza en la clasificación")

        st.balloons()
    else:
        st.info("👋 ¡Esperando tu imagen! Selecciona o arrastra una foto para comenzar.")

# --- Ejecutar la app ---
if __name__ == "__main__":
    main()
