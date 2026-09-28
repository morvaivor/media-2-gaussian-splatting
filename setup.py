#!/usr/bin/env python3
"""
Script d'installation du pipeline 3DGS Insta360.
Cross-platform : Windows + Linux.

Installe dans un environnement isolé (dossier dédié + venv).
Ne modifie PAS les logiciels existants (RealityScan, LichtFeld Studio, etc.).
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

INSTALL_DIR = Path(__file__).parent / "gs_env"
BIN_DIR = INSTALL_DIR / "bin"
PYTHON_DIR = INSTALL_DIR / "python"
VENV_DIR = INSTALL_DIR / "venv"

PYTHON_PACKAGES = [
    "opencv-python",
    "numpy",
    "ultralytics",
]

OPTIONAL_PACKAGES = {
    "sam2": ["segment-anything"],
    "torch": ["torch", "torchvision"],
}


def log(msg: str):
    print(f"[SETUP] {msg}")


def warn(msg: str):
    print(f"[SETUP] WARNING: {msg}", file=sys.stderr)


def error(msg: str):
    print(f"[SETUP] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def run_cmd(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    log(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        error(f"Commande échouée : {' '.join(cmd)}\n{result.stderr[:500]}")
    return result


def detect_gpu() -> dict:
    """Détecte le GPU et les drivers."""
    info = {"gpu": None, "nvidia": False, "cuda": False, "driver": None}

    # NVIDIA (Windows et Linux)
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines:
                parts = lines[0].split(", ")
                info["gpu"] = parts[0] if parts else "NVIDIA GPU"
                info["driver"] = parts[1] if len(parts) > 1 else None
                info["nvidia"] = True
                info["cuda"] = True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # AMD/Intel (Linux uniquement, si pas de NVIDIA)
    if not info["nvidia"] and IS_LINUX:
        try:
            result = subprocess.run(["lspci"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and "VGA" in result.stdout:
                for line in result.stdout.split("\n"):
                    if "VGA" in line or "3D" in line:
                        info["gpu"] = line.strip()
                        break
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    return info


def check_prerequisites():
    """Vérifie les prérequis système."""
    log("Vérification des prérequis...")

    # Python
    if sys.version_info < (3, 10):
        warn(f"Python {sys.version} détecté. Python 3.10+ recommandé.")

    # GPU
    gpu_info = detect_gpu()
    if gpu_info["nvidia"]:
        log(f"  GPU NVIDIA détecté : {gpu_info['gpu']}")
        if gpu_info["driver"]:
            log(f"  Driver : {gpu_info['driver']}")
    elif gpu_info["gpu"]:
        log(f"  GPU détecté : {gpu_info['gpu']}")
        warn("GPU non-NVIDIA détecté. Brush supporte AMD/Intel via WebGPU, mais COLMAP peut être plus lent.")
    else:
        warn("Aucun GPU détecté. Le training 3DGS sera très lent ou impossible.")

    # ffmpeg
    if shutil.which("ffmpeg") is None:
        warn("ffmpeg introuvable dans PATH. Il sera installé si possible.")
    else:
        log(f"  ffmpeg trouvé : {shutil.which('ffmpeg')}")

    # git
    if shutil.which("git") is None:
        warn("git introuvable dans PATH. Nécessaire pour cloner Brush.")

    return gpu_info


def install_ffmpeg():
    """Installe ffmpeg si absent."""
    if shutil.which("ffmpeg") is not None:
        log("  ffmpeg déjà installé.")
        return

    log("  Installation de ffmpeg...")
    if IS_WINDOWS:
        # Télécharger le binaire Windows
        url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        log(f"  Téléchargement de {url}...")
        # Utiliser curl ou wget
        try:
            run_cmd(["curl", "-L", "-o", str(INSTALL_DIR / "ffmpeg.zip"), url])
            run_cmd(["powershell", "-Command",
                     f"Expand-Archive -Path '{INSTALL_DIR / 'ffmpeg.zip'}' -DestinationPath '{BIN_DIR}' -Force"])
            # Trouver le binaire ffmpeg.exe
            for f in BIN_DIR.rglob("ffmpeg.exe"):
                log(f"  ffmpeg installé dans {f.parent}")
                return
        except Exception as e:
            warn(f"  Impossible d'installer ffmpeg automatiquement : {e}")
            warn("  Installez ffmpeg manuellement : https://ffmpeg.org/download.html")
    elif IS_LINUX:
        try:
            if shutil.which("apt-get"):
                run_cmd(["sudo", "apt-get", "install", "-y", "ffmpeg"])
            elif shutil.which("yum"):
                run_cmd(["sudo", "yum", "install", "-y", "ffmpeg"])
            elif shutil.which("dnf"):
                run_cmd(["sudo", "dnf", "install", "-y", "ffmpeg"])
            else:
                warn("  Package manager inconnu. Installez ffmpeg manuellement.")
        except Exception as e:
            warn(f"  Impossible d'installer ffmpeg : {e}")


def install_colmap():
    """Installe COLMAP (binaire précompilé)."""
    if shutil.which("colmap") is not None:
        log("  COLMAP déjà installé.")
        return

    log("  Installation de COLMAP...")
    if IS_WINDOWS:
        warn("  COLMAP n'a pas de binaire précompilé officiel pour Windows.")
        warn("  Installez COLMAP manuellement : https://colmap.github.io/install.html")
        warn("  Puis utilisez --colmap-path pour indiquer l'emplacement du binaire.")
    elif IS_LINUX:
        try:
            if shutil.which("apt-get"):
                log("  Tentative d'installation via apt-get...")
                run_cmd(["sudo", "apt-get", "install", "-y", "colmap"], check=False)
                if shutil.which("colmap") is not None:
                    log("  COLMAP installé via apt-get.")
                    return
            elif shutil.which("yum"):
                log("  Tentative d'installation via yum...")
                run_cmd(["sudo", "yum", "install", "-y", "colmap"], check=False)
                if shutil.which("colmap") is not None:
                    log("  COLMAP installé via yum.")
                    return
        except Exception as e:
            warn(f"  Impossible d'installer COLMAP automatiquement : {e}")
        warn("  Installez COLMAP manuellement : https://colmap.github.io/install.html")
        warn("  Puis utilisez --colmap-path pour indiquer l'emplacement du binaire.")


def install_brush():
    """Installe Brush (binaire précompilé ou build Rust)."""
    if shutil.which("brush") is not None:
        log("  Brush déjà installé.")
        return

    log("  Installation de Brush...")
    if shutil.which("cargo") is not None:
        log("  Rust (cargo) détecté, tentative de build depuis la source...")
        try:
            run_cmd(["git", "clone", "--depth", "1", "https://github.com/ArthurBrussee/brush.git",
                     str(INSTALL_DIR / "brush-src")])
            run_cmd(["cargo", "install", "--path", str(INSTALL_DIR / "brush-src"), "--force"], check=False)
            if shutil.which("brush") is not None:
                log("  Brush installé via cargo.")
                return
        except Exception as e:
            warn(f"  Build de Brush échoué : {e}")
    else:
        warn("  Rust (cargo) non détecté. Impossible de builder Brush depuis la source.")

    warn("  Installez Brush manuellement : https://github.com/ArthurBrussee/brush")
    warn("  Puis utilisez --brush-path pour indiquer l'emplacement du binaire.")


def setup_venv():
    """Crée un environnement virtuel Python isolé."""
    log("  Création de l'environnement virtuel Python...")
    if VENV_DIR.exists():
        log("  venv déjà existant, réutilisation...")
        return

    venv_script = str(VENV_DIR / ("Scripts" if IS_WINDOWS else "bin") / "python")
    run_cmd([sys.executable, "-m", "venv", str(VENV_DIR)])

    pip_path = str(VENV_DIR / ("Scripts" if IS_WINDOWS else "bin") / "pip")
    python_path = venv_script

    # Upgrader pip
    run_cmd([python_path, "-m", "pip", "install", "--upgrade", "pip"], check=False)

    # Installer les packages
    log("  Installation des packages Python...")
    for pkg in PYTHON_PACKAGES:
        run_cmd([python_path, "-m", "pip", "install", pkg], check=False)

    # PyTorch avec CUDA
    log("  Installation de PyTorch (CUDA)...")
    if IS_WINDOWS:
        run_cmd([python_path, "-m", "pip", "install", "torch", "torchvision",
                 "--index-url", "https://download.pytorch.org/whl/cu118"], check=False)
    else:
        run_cmd([python_path, "-m", "pip", "install", "torch", "torchvision",
                 "--index-url", "https://download.pytorch.org/whl/cu118"], check=False)

    log("  Environnement virtuel prêt.")


def main():
    log("=" * 60)
    log("INSTALLATION PIPELINE 3DGS INSTA360")
    log("=" * 60)
    log(f"Système : {platform.system()} {platform.release()}")
    log(f"Python : {sys.version}")
    log(f"Dossier d'installation : {INSTALL_DIR}")
    log("")

    # Vérifier les prérequis
    gpu_info = check_prerequisites()
    log("")

    # Créer les dossiers
    log("Création des dossiers...")
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    log(f"  {INSTALL_DIR}")
    log("")

    # Installer ffmpeg
    log("Étape 1/4 : ffmpeg")
    install_ffmpeg()
    log("")

    # Installer COLMAP
    log("Étape 2/4 : COLMAP")
    install_colmap()
    log("")

    # Installer Brush
    log("Étape 3/4 : Brush")
    install_brush()
    log("")

    # Setup venv Python
    log("Étape 4/4 : Environnement Python")
    setup_venv()
    log("")

    # Résumé
    log("=" * 60)
    log("RÉSUMÉ DE L'INSTALLATION")
    log("=" * 60)
    log(f"  Dossier d'installation : {INSTALL_DIR}")
    log(f"  Binaires : {BIN_DIR}")
    log(f"  venv Python : {VENV_DIR}")
    log("")
    log("  Pour activer l'environnement :")
    if IS_WINDOWS:
        log(f"    {VENV_DIR / 'Scripts' / 'activate'}")
    else:
        log(f"    source {VENV_DIR / 'bin' / 'activate'}")
    log("")
    log("  Pour utiliser le pipeline :")
    log("    1. Activez l'environnement virtuel (voir ci-dessus)")
    log("    2. python gs_pipeline.py run-all --input ma_video.mp4")
    log("")
    log("  NOTE : Cette installation n'a pas modifié les logiciels existants")
    log("  (RealityScan, LichtFeld Studio, etc.).")
    log("=" * 60)


if __name__ == "__main__":
    main()
