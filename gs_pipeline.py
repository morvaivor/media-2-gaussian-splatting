#!/usr/bin/env python3
"""
Pipeline 3D Gaussian Splatting depuis vidéo/photo Insta360 X4.

Étapes :
  1. preprocess  - Extraction frames + déstériscoping 360° → 8 vues perspective
  2. mask        - Masquage cameraman (YOLOv8 / SAM2)
  3. reconstruct - Reconstruction COLMAP
  4. train       - Training 3DGS avec Brush
  5. run-all     - Toutes les étapes séquentielles
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

VIDEO_EXTS = {".mp4", ".insv", ".mov", ".avi", ".mkv"}
PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class PipelineError(Exception):
    pass


class PipelineLog:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.start_time = datetime.now().isoformat()
        self.steps = []
        self.errors = []
        self.warnings = []
        self.params = {}
        self.image_count = 0
        self.output_ply = None
        self.end_time = None

    def add_step(self, name: str, duration: float, details: dict = None):
        step = {
            "name": name,
            "start": datetime.now().isoformat(),
            "duration_seconds": round(duration, 2),
            "status": "success",
        }
        if details:
            step["details"] = details
        self.steps.append(step)
        print(f"  [{name}] OK ({duration:.1f}s)")

    def add_error(self, step: str, message: str):
        self.errors.append({"step": step, "message": message})
        print(f"  ERROR [{step}]: {message}", file=sys.stderr)

    def add_warning(self, message: str):
        self.warnings.append(message)
        print(f"  WARNING: {message}", file=sys.stderr)

    def finish(self, output_ply: str = None):
        self.end_time = datetime.now().isoformat()
        self.output_ply = output_ply
        data = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_seconds": round(
                (datetime.fromisoformat(self.end_time) - datetime.fromisoformat(self.start_time)).total_seconds(), 2
            ),
            "steps": self.steps,
            "errors": self.errors,
            "warnings": self.warnings,
            "params": self.params,
            "image_count": self.image_count,
            "output_ply": self.output_ply,
        }
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nLog écrit dans {self.log_path}")

    def to_dict(self):
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "steps": self.steps,
            "errors": self.errors,
            "warnings": self.warnings,
            "params": self.params,
            "image_count": self.image_count,
            "output_ply": self.output_ply,
        }


def check_binary(name: str, hint: str = None) -> bool:
    if shutil.which(name) is None:
        msg = f"Binaire '{name}' introuvable dans PATH."
        if hint:
            msg += f" {hint}"
        raise PipelineError(msg)
    return True


def check_ffmpeg():
    check_binary("ffmpeg", "Installez ffmpeg : https://ffmpeg.org/download.html")


def check_opencv():
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        raise PipelineError("OpenCV (cv2) et/ou numpy non installés. Installez-les avec : pip install opencv-python numpy")


def check_python_deps():
    check_opencv()
    try:
        import torch  # noqa: F401
    except ImportError:
        raise PipelineError("PyTorch non installé. Installez-le avec : pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")


def find_input_files(input_path: Path):
    videos = []
    photos = []

    if input_path.is_file():
        ext = input_path.suffix.lower()
        if ext in VIDEO_EXTS:
            videos.append(input_path)
        elif ext in PHOTO_EXTS:
            photos.append(input_path)
        else:
            raise PipelineError(f"Format non reconnu : {ext}. Formats acceptés : {VIDEO_EXTS | PHOTO_EXTS}")
    elif input_path.is_dir():
        for f in sorted(input_path.iterdir()):
            if f.is_file():
                ext = f.suffix.lower()
                if ext in VIDEO_EXTS:
                    videos.append(f)
                elif ext in PHOTO_EXTS:
                    photos.append(f)
    else:
        raise PipelineError(f"Chemin d'entrée introuvable : {input_path}")

    if not videos and not photos:
        raise PipelineError(f"Aucun fichier vidéo ou photo trouvé dans {input_path}")

    return videos, photos





def extract_video_frames(video_path: Path, output_dir: Path, fps: int = 2) -> list:
    """Extrait des frames d'une vidéo à la fréquence donnée.
    Pour les .insv, essaie d'abord Insta360 Studio, puis ffmpeg.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(output_dir / "frame_%05d.jpg")

    # Pour les .insv, essayer Insta360 Studio d'abord
    if video_path.suffix.lower() == ".insv":
        insta360_paths = [
            r"C:\Program Files\Insta360\Studio\Studio.exe",
            r"C:\Program Files (x86)\Insta360\Studio\Studio.exe",
            "/opt/Insta360/Studio/Studio",
            "/usr/local/bin/insta360-studio",
        ]
        for p in insta360_paths:
            if os.path.exists(p):
                try:
                    cmd = [p, "--convert", str(video_path), "--output", str(output_dir), "--format", "equirectangular"]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    if result.returncode == 0:
                        frames = sorted(output_dir.glob("*.jpg")) + sorted(output_dir.glob("*.png"))
                        if frames:
                            print(f"    {len(frames)} frames extraites via Insta360 Studio")
                            return frames
                except (subprocess.TimeoutExpired, OSError):
                    pass
        print("    Insta360 Studio non trouvé, utilisation de ffmpeg pour .insv...")

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        pattern,
    ]
    print(f"  Extraction des frames de {video_path.name} à {fps} FPS...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(f"ffmpeg a échoué lors de l'extraction des frames : {result.stderr[:500]}")

    frames = sorted(output_dir.glob("frame_*.jpg"))
    if not frames:
        raise PipelineError(f"Aucune frame extraite de {video_path.name}")
    print(f"    {len(frames)} frames extraites")
    return frames


def destripe_360_to_perspective(input_path: Path, output_dir: Path, num_views: int = 8, fov: float = 90.0, width: int = 1920, height: int = 1080) -> list:
    """Convertis une image 360° equirectangular en N vues perspective.
    Pour chaque vue, recalcule les coordonnées u ET v à partir du vecteur direction roté.
    """
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(input_path))
    if img is None:
        raise PipelineError(f"Impossible de lire l'image : {input_path}")

    h, w = img.shape[:2]
    # L'image equirectangular doit avoir un ratio 2:1
    if w != h * 2:
        img = cv2.resize(img, (h * 2, h))
        w = h * 2

    fov_rad = np.deg2rad(fov)
    fx = (width / 2) / np.tan(fov_rad / 2)
    cx, cy = width / 2, height / 2

    # Grille de pixels pour la projection (vecteurs direction unitaires)
    u = np.arange(width)
    v = np.arange(height)
    u_grid, v_grid = np.meshgrid(u, v)
    x = (u_grid - cx) / fx
    y = (v_grid - cy) / fx
    z = np.ones_like(x)
    norm = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x / norm, y / norm, z / norm

    output_paths = []
    for i in range(num_views):
        yaw = (2 * np.pi * i) / num_views
        # Rotation du vecteur direction selon le yaw
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        x_rot = x * cos_yaw - z * sin_yaw
        z_rot = x * sin_yaw + z * cos_yaw
        # y est invariant pour une rotation de yaw pure
        # Renormaliser pour s'assurer que le vecteur est unitaire
        norm_rot = np.sqrt(x_rot**2 + y**2 + z_rot**2)
        x_rot, y_rot, z_rot = x_rot / norm_rot, y / norm_rot, z_rot / norm_rot

        # Conversion vers coordonnées sphériques DU VECTEUR ROTÉ
        theta_rot = np.arctan2(x_rot, z_rot)  # longitude [-pi, pi]
        phi_rot = np.arcsin(np.clip(y_rot, -1, 1))  # latitude [-pi/2, pi/2]

        # Mapping vers les coordonnées de l'image equirectangular
        u_eq_rot = ((theta_rot + np.pi) / (2 * np.pi)) * w
        v_eq_rot = ((phi_rot + np.pi / 2) / np.pi) * h
        u_eq_rot = np.clip(u_eq_rot.astype(np.int32), 0, w - 1)
        v_eq_rot = np.clip(v_eq_rot.astype(np.int32), 0, h - 1)

        # Échantillonnage bilinéaire pour une meilleure qualité
        perspective_img = cv2.remap(
            img,
            u_eq_rot.astype(np.float32),
            v_eq_rot.astype(np.float32),
            cv2.INTER_LINEAR,
        )
        out_path = output_dir / f"view_{i:02d}.jpg"
        cv2.imwrite(str(out_path), perspective_img)
        output_paths.append(out_path)

    return output_paths


def preprocess(args):
    """Étape 1 : Prétraitement vidéo et photos."""
    print("\n=== ÉTAPE 1 : PRÉTRAITEMENT ===")
    start = time.time()
    check_ffmpeg()
    check_opencv()

    work_dir = Path(args.work_dir)
    frames_dir = work_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    videos, photos = find_input_files(Path(args.input))
    total_images = 0

    # Traitement des vidéos (y compris .insv)
    for video in videos:
        video_frames_dir = frames_dir / f"video_{video.stem}"
        frames = extract_video_frames(video, video_frames_dir, fps=args.fps)
        for frame in frames:
            view_dir = frames_dir / f"video_{video.stem}" / f"{frame.stem}"
            views = destripe_360_to_perspective(
                frame, view_dir,
                num_views=args.num_views,
                fov=args.fov,
                width=args.width,
                height=args.height,
            )
            total_images += len(views)
        print(f"    Vidéo {video.name} : {len(frames)} frames → {len(frames) * args.num_views} vues")

    # Traitement des photos 360° equirectangular
    for photo in photos:
        view_dir = frames_dir / f"{photo.stem}"
        views = destripe_360_to_perspective(
            photo, view_dir,
            num_views=args.num_views,
            fov=args.fov,
            width=args.width,
            height=args.height,
        )
        total_images += len(views)
        print(f"    Photo {photo.name} : {len(views)} vues")

    duration = time.time() - start
    return {
        "total_images": total_images,
        "videos": len(videos),
        "photos": len(photos),
        "duration": duration,
    }


def mask_cameraman(args):
    """Étape 2 : Masquage du cameraman."""
    print("\n=== ÉTAPE 2 : MASQUAGE CAMERAMAN ===")
    start = time.time()
    check_opencv()
    import cv2

    work_dir = Path(args.work_dir)
    frames_dir = work_dir / "frames"
    masked_dir = work_dir / "frames_masked"
    masks_dir = work_dir / "masks"
    masked_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    if not frames_dir.exists():
        raise PipelineError(f"Dossier frames/ introuvable : {frames_dir}. Lancez d'abord l'étape preprocess.")

    # Collecte de toutes les images
    all_images = sorted(frames_dir.rglob("*.jpg")) + sorted(frames_dir.rglob("*.png"))
    if not all_images:
        raise PipelineError(f"Aucune image trouvée dans {frames_dir}")

    # Masque manuel
    if args.mask_path:
        mask = cv2.imread(str(args.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise PipelineError(f"Impossible de lire le masque : {args.mask_path}")
        print(f"  Utilisation du masque manuel : {args.mask_path}")
    else:
        print("  Détection avec YOLOv8...")
        try:
            from ultralytics import YOLO
        except ImportError:
            raise PipelineError("YOLOv8 (ultralytics) non installé. Installez-le avec : pip install ultralytics")
        print("  NOTE : Le modèle YOLOv8 sera téléchargé au premier lancement (~6 MB).")
        model = YOLO("yolov8n.pt")

    count = 0
    for img_path in all_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        if args.mask_path:
            mask_resized = cv2.resize(mask, (w, h))
            binary_mask = (mask_resized > 128).astype(np.uint8) * 255
        else:
            results = model(img, verbose=False)
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            for r in results:
                for box in r.boxes:
                    if int(r.names[box.cls[0]]) == 0:  # person class
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        binary_mask[y1:y2, x1:x2] = 255

        # Appliquer le masque : pixels masqués → noir
        masked_img = img.copy()
        masked_img[binary_mask > 0] = 0

        # Sauvegarder dans la même structure de dossiers
        rel_path = img_path.relative_to(frames_dir)
        out_path = masked_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), masked_img)

        # Sauvegarder le masque (noir = ignoré, blanc = conservé → inverser)
        mask_out = 255 - binary_mask  # blanc = conservé, noir = ignoré
        mask_out_path = masks_dir / rel_path.with_suffix(".png")
        mask_out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_out_path), mask_out)

        count += 1

    duration = time.time() - start
    return {"images_masked": count, "duration": duration}


def reconstruct(args):
    """Étape 3 : Reconstruction COLMAP."""
    print("\n=== ÉTAPE 3 : RECONSTRUCTION COLMAP ===")
    start = time.time()

    work_dir = Path(args.work_dir)
    masked_dir = work_dir / "frames_masked"
    colmap_dir = work_dir / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    if not masked_dir.exists():
        raise PipelineError(f"Dossier frames_masked/ introuvable : {masked_dir}. Lancez d'abord l'étape mask.")

    # Trouver le binaire COLMAP
    colmap_bin = None
    if args.colmap_path:
        colmap_bin = args.colmap_path
    else:
        colmap_bin = shutil.which("colmap")
        if colmap_bin is None:
            # Chercher dans des emplacements courants
            candidates = [
                r"C:\Program Files\COLMAP\bin\colmap.exe",
                r"C:\COLMAP\bin\colmap.exe",
                "/usr/local/bin/colmap",
                "/usr/bin/colmap",
            ]
            for c in candidates:
                if os.path.exists(c):
                    colmap_bin = c
                    break
        if colmap_bin is None:
            raise PipelineError("COLMAP introuvable. Installez COLMAP ou utilisez --colmap-path.")

    # Structure COLMAP attendue
    colmap_images = colmap_dir / "images"
    colmap_sparse = colmap_dir / "sparse" / "0"
    colmap_images.mkdir(parents=True, exist_ok=True)
    colmap_sparse.mkdir(parents=True, exist_ok=True)

    # Copier les images masquées
    print("  Copie des images...")
    for img in sorted(masked_dir.rglob("*.jpg")):
        rel = img.relative_to(masked_dir)
        dest = colmap_images / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(img), str(dest))

    # Feature extraction (auto camera model)
    print("  Feature extraction...")
    cmd_extract = [
        colmap_bin, "feature_extractor",
        "--database_path", str(colmap_dir / "database.db"),
        "--image_path", str(colmap_images),
    ]
    result = subprocess.run(cmd_extract, capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(f"COLMAP feature_extractor a échoué : {result.stderr[:500]}")

    # Feature matching (séquentiel pour les frames vidéo)
    print("  Feature matching (séquentiel)...")
    cmd_match = [
        colmap_bin, "sequential_matcher",
        "--database_path", str(colmap_dir / "database.db"),
        "--loop_detection", "0",
    ]
    result = subprocess.run(cmd_match, capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(f"COLMAP sequential_matcher a échoué : {result.stderr[:500]}")

    # Bundle adjustment / reconstruction incrémentale
    print("  Reconstruction incrémentale...")
    cmd_recon = [
        colmap_bin, "mapper",
        "--database_path", str(colmap_dir / "database.db"),
        "--image_path", str(colmap_images),
        "--output_path", str(colmap_dir / "sparse"),
    ]
    result = subprocess.run(cmd_recon, capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(f"COLMAP mapper a échoué : {result.stderr[:500]}")

    # Vérifier que les fichiers de sortie existent
    required_files = ["cameras.bin", "images.bin", "points3D.bin"]
    for f in required_files:
        if not (colmap_sparse / f).exists():
            raise PipelineError(f"Fichier COLMAP manquant : {colmap_sparse / f}")

    duration = time.time() - start
    return {"duration": duration}


def train(args):
    """Étape 4 : Training 3DGS avec Brush."""
    print("\n=== ÉTAPE 4 : TRAINING 3DGS (BRUSH) ===")
    start = time.time()

    work_dir = Path(args.work_dir)
    colmap_dir = work_dir / "colmap"
    masks_dir = work_dir / "masks"
    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not colmap_dir.exists():
        raise PipelineError(f"Dossier colmap/ introuvable : {colmap_dir}. Lancez d'abord l'étape reconstruct.")

    # Trouver le binaire Brush
    brush_bin = None
    if args.brush_path:
        brush_bin = args.brush_path
    else:
        brush_bin = shutil.which("brush")
        if brush_bin is None:
            candidates = [
                r"C:\Program Files\Brush\brush.exe",
                r"C:\Brush\brush.exe",
                "/usr/local/bin/brush",
                "/usr/bin/brush",
            ]
            for c in candidates:
                if os.path.exists(c):
                    brush_bin = c
                    break
        if brush_bin is None:
            raise PipelineError("Brush introuvable. Installez Brush ou utilisez --brush-path.")

    # Structure COLMAP pour Brush
    # Brush attend : images/ + sparse/0/{cameras.bin, images.bin, points3D.bin}
    # + optionnellement masks/

    # Copier les masques dans le dossier COLMAP si ils existent
    if masks_dir.exists():
        colmap_masks = colmap_dir / "masks"
        if not colmap_masks.exists():
            shutil.copytree(str(masks_dir), str(colmap_masks))

    # Lancer Brush
    print(f"  Lancement de Brush (iterations={args.iterations})...")
    cmd = [
        brush_bin, "train",
        "--data", str(colmap_dir),
        "--iterations", str(args.iterations),
    ]
    if args.resolution:
        cmd.extend(["--resolution", str(args.resolution)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(f"Brush a échoué : {result.stderr[:500]}")

    # Brush écrit le .ply dans le dossier de données
    # Chercher le fichier .ply généré
    model_ply = None
    for ply in colmap_dir.rglob("*.ply"):
        model_ply = ply
        break
    if model_ply is None:
        raise PipelineError(f"Fichier .ply non généré dans {colmap_dir}")

    # Copier le .ply dans le dossier output
    shutil.copy2(str(model_ply), str(output_dir / "model.ply"))
    model_ply = output_dir / "model.ply"

    duration = time.time() - start
    return {"model_ply": str(model_ply), "duration": duration}


def run_all(args):
    """Exécute toutes les étapes séquentiellement."""
    log = PipelineLog(Path(args.work_dir) / "pipeline_log.json")
    log.params = {
        "input": args.input,
        "fps": args.fps,
        "num_views": args.num_views,
        "fov": args.fov,
        "width": args.width,
        "height": args.height,
        "mask_path": args.mask_path,
        "iterations": args.iterations,
        "resolution": args.resolution,
    }

    steps = [
        ("preprocess", preprocess),
        ("mask", mask_cameraman),
        ("reconstruct", reconstruct),
        ("train", train),
    ]

    for step_name, step_func in steps:
        try:
            result = step_func(args)
            duration = result.pop("duration", 0)
            log.add_step(step_name, duration, result)
            if step_name == "preprocess":
                log.image_count = result.get("total_images", 0)
            if step_name == "train":
                log.output_ply = result.get("model_ply")
        except PipelineError as e:
            log.add_error(step_name, str(e))
            log.finish()
            print(f"\nPipeline arrêté à l'étape '{step_name}' : {e}")
            sys.exit(1)
        except Exception as e:
            log.add_error(step_name, f"Erreur inattendue : {e}")
            log.finish()
            print(f"\nPipeline arrêté à l'étape '{step_name}' : {e}")
            sys.exit(1)

    log.finish()
    print(f"\n=== PIPELINE TERMINÉ ===")
    if log.output_ply:
        print(f"Modèle 3DGS : {log.output_ply}")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline 3D Gaussian Splatting depuis vidéo/photo Insta360 X4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  %(prog)s run-all --input ma_video.mp4
  %(prog)s run-all --input mon_dossier/
  %(prog)s preprocess --input ma_video.mp4 --fps 5
  %(prog)s mask
  %(prog)s reconstruct
  %(prog)s train --iterations 30000
        """,
    )
    parser.add_argument("command", choices=["preprocess", "mask", "reconstruct", "train", "run-all"],
                        help="Étape à exécuter")
    parser.add_argument("--input", "-i", default=None,
                        help="Chemin d'entrée (vidéo, photo, ou dossier)")
    parser.add_argument("--work-dir", "-w", default=".",
                        help="Dossier de travail (défaut : .)")
    parser.add_argument("--fps", type=int, default=2,
                        help="Fréquence d'extraction des frames (défaut : 2)")
    parser.add_argument("--num-views", type=int, default=8,
                        help="Nombre de vues perspective par frame 360° (défaut : 8)")
    parser.add_argument("--fov", type=float, default=90.0,
                        help="Champ de vision en degrés (défaut : 90)")
    parser.add_argument("--width", type=int, default=1920,
                        help="Largeur de sortie (défaut : 1920)")
    parser.add_argument("--height", type=int, default=1080,
                        help="Hauteur de sortie (défaut : 1080)")
    parser.add_argument("--mask-path", default=None,
                        help="Chemin vers un masque manuel (image noir/blanc)")
    parser.add_argument("--colmap-path", default=None,
                        help="Chemin vers le binaire COLMAP")
    parser.add_argument("--brush-path", default=None,
                        help="Chemin vers le binaire Brush")
    parser.add_argument("--iterations", type=int, default=30000,
                        help="Nombre d'itérations de training (défaut : 30000)")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Résolution de training (optionnel)")

    args = parser.parse_args()

    # Vérifier que --input est fourni pour preprocess et run-all
    if args.command in ("preprocess", "run-all") and not args.input:
        parser.error("--input est requis pour les commandes 'preprocess' et 'run-all'")

    try:
        if args.command == "preprocess":
            result = preprocess(args)
            print(f"\nPrétraitement terminé : {result['total_images']} images")
        elif args.command == "mask":
            result = mask_cameraman(args)
            print(f"\nMasquage terminé : {result['images_masked']} images")
        elif args.command == "reconstruct":
            result = reconstruct(args)
            print(f"\nReconstruction COLMAP terminée")
        elif args.command == "train":
            result = train(args)
            print(f"\nTraining terminé : {result['model_ply']}")
        elif args.command == "run-all":
            run_all(args)
    except PipelineError as e:
        print(f"\nERREUR : {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
