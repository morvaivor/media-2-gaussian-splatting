# Pipeline 3D Gaussian Splatting - Insta360 X4

Pipeline automatisé pour créer un modèle 3D Gaussian Splatting à partir de vidéos et photos 360° capturées avec une caméra Insta360 X4.

## Fonctionnalités

- **Prétraitement** : extraction de frames vidéo (2 FPS par défaut) + déstériscoping 360° equirectangular → 8 vues perspective (FOV 90°)
- **Support multi-format** : vidéos (.mp4, .insv, .mov, .avi, .mkv) et photos (.jpg, .png, .tif)
- **Masquage cameraman** : détection automatique avec YOLOv8 (par défaut) ou masque manuel
- **Reconstruction 3D** : COLMAP (feature extraction + matching + bundle adjustment)
- **Training 3DGS** : Brush (open source, cross-platform, AMD/NVIDIA/Intel)
- **Log JSON** : rapport détaillé de chaque exécution
- **Cross-platform** : Windows + Linux

## Prérequis

- **Python** 3.10+
- **GPU** : NVIDIA (recommandé) ou AMD/Intel (via WebGPU pour Brush)
- **ffmpeg** : pour l'extraction de frames et la conversion vidéo
- **COLMAP** : pour la reconstruction 3D (binaire précompilé)
- **Brush** : pour le training 3DGS (binaire précompilé)
- **~10 GB** d'espace disque pour l'installation

## Installation

### Option 1 : Script automatique (recommandé)

```bash
python setup.py
```

Le script :
1. Détecte l'OS et le GPU
2. Installe ffmpeg, COLMAP, Brush dans `gs_env/bin/`
3. Crée un environnement virtuel Python dans `gs_env/venv/`
4. Installe les packages Python (OpenCV, PyTorch, ultralytics)

**Important** : L'installation est isolée dans `gs_env/` et ne modifie PAS les logiciels existants (RealityScan, LichtFeld Studio, etc.).

### Option 2 : Installation manuelle

1. **ffmpeg** : https://ffmpeg.org/download.html
2. **COLMAP** : https://colmap.github.io/install.html
3. **Brush** : https://github.com/ArthurBrussee/brush (build avec Rust/cargo)
4. **Python venv** :
   ```bash
   python -m venv gs_env/venv
   # Windows :
   gs_env\venv\Scripts\activate
   # Linux :
   source gs_env/venv/bin/activate
   pip install -r requirements.txt
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

## Utilisation

### Pipeline complet (recommandé)

```bash
# Vidéo seule
python gs_pipeline.py run-all --input ma_video.mp4

# Dossier avec vidéo + photos
python gs_pipeline.py run-all --input mon_dossier/

# Avec options
python gs_pipeline.py run-all --input ma_video.mp4 --fps 5 --num-views 12 --iterations 50000
```

### Étape par étape

```bash
# 1. Prétraitement (extraction frames + déstériscoping)
python gs_pipeline.py preprocess --input ma_video.mp4 --fps 2 --num-views 8

# 2. Masquage cameraman
python gs_pipeline.py mask                    # YOLOv8 (par défaut)
python gs_pipeline.py mask --mask-path masque.png  # Masque manuel

# 3. Reconstruction COLMAP
python gs_pipeline.py reconstruct

# 4. Training 3DGS (Brush)
python gs_pipeline.py train --iterations 30000
```

### Paramètres

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `--input`, `-i` | - | Chemin d'entrée (vidéo, photo, ou dossier) |
| `--work-dir`, `-w` | `.` | Dossier de travail |
| `--fps` | 2 | Fréquence d'extraction des frames |
| `--num-views` | 8 | Nombre de vues perspective par frame 360° |
| `--fov` | 90 | Champ de vision en degrés |
| `--width` | 1920 | Largeur de sortie |
| `--height` | 1080 | Hauteur de sortie |
| `--mask-path` | - | Chemin vers un masque manuel |
| `--colmap-path` | - | Chemin vers le binaire COLMAP |
| `--brush-path` | - | Chemin vers le binaire Brush |
| `--iterations` | 30000 | Nombre d'itérations de training |
| `--resolution` | - | Résolution de training (optionnel) |

## Structure des dossiers

```
projet/
├── gs_pipeline.py          # CLI principal
├── setup.py                # Installation
├── uninstall.py            # Désinstallation
├── requirements.txt        # Dépendances Python
├── README.md               # Ce fichier
├── gs_env/                 # Environnement isolé (créé par setup.py)
│   ├── bin/                # Binaires (COLMAP, Brush, ffmpeg)
│   └── venv/               # Environnement virtuel Python
├── frames/                 # Frames extraites + vues perspective (étape 1)
│   ├── video_ma_video/
│   │   ├── frame_00001/
│   │   │   ├── view_00.jpg
│   │   │   ├── view_01.jpg
│   │   │   └── ...
│   │   └── ...
│   └── photo1/
│       ├── view_00.jpg
│       └── ...
├── frames_masked/          # Images masquées (étape 2)
├── masks/                  # Masques (noir = ignoré, blanc = conservé)
├── colmap/                 # Données COLMAP (étape 3)
│   ├── images/
│   ├── sparse/0/
│   │   ├── cameras.bin
│   │   ├── images.bin
│   │   └── points3D.bin
│   └── database.db
├── output/                 # Résultat (étape 4)
│   └── model.ply           # Modèle 3D Gaussian Splatting
└── pipeline_log.json       # Log de l'exécution
```

## Log JSON

Chaque exécution génère `pipeline_log.json` :

```json
{
  "start_time": "2026-09-28T10:00:00",
  "end_time": "2026-09-28T12:30:00",
  "total_duration_seconds": 9000.0,
  "steps": [
    {"name": "preprocess", "duration_seconds": 120.5, "status": "success"},
    {"name": "mask", "duration_seconds": 300.2, "status": "success"},
    {"name": "reconstruct", "duration_seconds": 1800.1, "status": "success"},
    {"name": "train", "duration_seconds": 6779.2, "status": "success"}
  ],
  "errors": [],
  "warnings": [],
  "params": {"input": "ma_video.mp4", "fps": 2, "iterations": 30000},
  "image_count": 960,
  "output_ply": "output/model.ply"
}
```

## Désinstallation

```bash
python uninstall.py
```

Le script :
1. Supprime les binaires installés par `setup.py` (COLMAP, Brush, ffmpeg)
2. Supprime l'environnement virtuel Python
3. Supprime les dossiers de données (frames/, colmap/, output/, etc.)
4. Demande confirmation avant chaque étape

**Important** : Ne supprime PAS les logiciels existants (RealityScan, LichtFeld Studio, etc.).

## Visualisation du résultat

Le fichier `output/model.ply` peut être visualisé avec :
- **Brush** : `brush output/model.ply`
- **SIBR Viewer** (fourni avec 3DGS officiel)
- **SuperSplat** : https://supersplat.atlas.engineer/
- **LichtFeld Studio**
- **Postshot**

## Troubleshooting

### ffmpeg introuvable
```bash
# Windows
# Téléchargez depuis https://www.gyan.dev/ffmpeg/builds/
# Ajoutez le dossier bin/ au PATH

# Linux
sudo apt-get install ffmpeg
```

### COLMAP échoue
- Vérifiez que les images sont bien dans `colmap/images/`
- Vérifiez que `colmap/sparse/0/` contient `cameras.bin`, `images.bin`, `points3D.bin`
- Essayez avec moins d'images : `--fps 1`

### Brush échoue
- Vérifiez que le GPU est compatible (NVIDIA RTX 2060+ ou AMD/Intel avec WebGPU)
- Vérifiez les drivers GPU
- Essayez avec moins d'itérations : `--iterations 10000`

### GPU insuffisant (VRAM)
- Réduisez la résolution : `--width 1280 --height 720`
- Réduisez le nombre de frames : `--fps 1`
- Réduisez le nombre d'itérations : `--iterations 15000`

### Conversion .insv échoue
- Installez Insta360 Studio : https://www.insta360.com/product/insta360-studio
- Ou vérifiez que ffmpeg supporte le format .insv

### YOLOv8 ne détecte pas le cameraman
- Utilisez un masque manuel : `--mask-path masque.png`
- Le masque doit être une image noir/blanc de la même taille que les frames
- Noir = zone à ignorer (cameraman), Blanc = zone à conserver

## Licences

- **gs_pipeline.py** : MIT
- **Brush** : Apache-2.0 (https://github.com/ArthurBrussee/brush)
- **COLMAP** : GPL-3.0 (https://colmap.github.io/)
- **YOLOv8** : AGPL-3.0 (https://github.com/ultralytics/ultralytics)
