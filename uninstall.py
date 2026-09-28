#!/usr/bin/env python3
"""
Script de désinstallation du pipeline 3DGS Insta360.
Cross-platform : Windows + Linux.

Ne supprime QUE ce qui a été installé par setup.py.
Ne touche PAS aux logiciels existants (RealityScan, LichtFeld Studio, etc.).
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
VENV_DIR = INSTALL_DIR / "venv"
DATA_DIRS = ["frames", "frames_masked", "masks", "colmap", "output"]


def log(msg: str):
    print(f"[UNINSTALL] {msg}")


def warn(msg: str):
    print(f"[UNINSTALL] WARNING: {msg}", file=sys.stderr)


def confirm(message: str) -> bool:
    """Demande confirmation à l'utilisateur."""
    try:
        response = input(f"{message} [o/N] : ").strip().lower()
        return response in ("o", "y", "oui", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Désinstallation du pipeline 3DGS Insta360")
    parser.add_argument("--work-dir", "-w", default=".",
                        help="Dossier de travail (défaut : .)")
    args = parser.parse_args()
    work_dir = Path(args.work_dir)

    log("=" * 60)
    log("DÉSINSTALLATION PIPELINE 3DGS INSTA360")
    log("=" * 60)
    log(f"Système : {platform.system()} {platform.release()}")
    log(f"Dossier de travail : {work_dir}")
    log("")

    removed = []

    # 1. Supprimer les binaires (COLMAP, Brush, ffmpeg installés par setup.py)
    if BIN_DIR.exists():
        log("Étape 1/4 : Suppression des binaires installés par setup.py")
        log(f"  Dossier : {BIN_DIR}")
        if confirm("  Supprimer les binaires (COLMAP, Brush, ffmpeg) ?"):
            shutil.rmtree(str(BIN_DIR))
            removed.append("binaires (COLMAP, Brush, ffmpeg)")
            log("  Binaires supprimés.")
        else:
            warn("  Binaires conservés.")
    else:
        log("Étape 1/4 : Aucun dossier de binaires trouvé.")
    log("")

    # 2. Supprimer le venv Python
    if VENV_DIR.exists():
        log("Étape 2/4 : Suppression de l'environnement virtuel Python")
        log(f"  Dossier : {VENV_DIR}")
        if confirm("  Supprimer l'environnement virtuel Python (PyTorch, OpenCV, ultralytics) ?"):
            shutil.rmtree(str(VENV_DIR))
            removed.append("environnement virtuel Python")
            log("  venv supprimé.")
        else:
            warn("  venv conservé.")
    else:
        log("Étape 2/4 : Aucun venv trouvé.")
    log("")

    # 3. Supprimer les dossiers de données
    log("Étape 3/4 : Suppression des dossiers de données")
    for d in DATA_DIRS:
        data_path = work_dir / d
        if data_path.exists():
            log(f"  Dossier : {data_path}")
            if confirm(f"  Supprimer {data_path} ?"):
                shutil.rmtree(str(data_path))
                removed.append(f"dossier {d}/")
                log(f"  {d}/ supprimé.")
            else:
                warn(f"  {d}/ conservé.")
    log("")

    # 4. Supprimer le fichier pipeline_log.json s'il existe
    log_file = work_dir / "pipeline_log.json"
    if log_file.exists():
        if confirm(f"  Supprimer {log_file} ?"):
            log_file.unlink()
            removed.append("pipeline_log.json")
            log("  pipeline_log.json supprimé.")
    log("")

    # 5. Supprimer le dossier gs_env s'il est vide
    if INSTALL_DIR.exists():
        try:
            # Vérifier s'il reste des fichiers
            remaining = list(INSTALL_DIR.iterdir())
            if not remaining:
                INSTALL_DIR.rmdir()
                removed.append("dossier gs_env/")
                log("Dossier gs_env/ supprimé (vide).")
            else:
                log(f"Dossier gs_env/ conservé (contient encore : {', '.join(f.name for f in remaining)})")
        except OSError:
            pass

    # Résumé
    log("=" * 60)
    log("RÉSUMÉ DE LA DÉSINSTALLATION")
    log("=" * 60)
    if removed:
        log("  Supprimé :")
        for item in removed:
            log(f"    - {item}")
    else:
        log("  Rien n'a été supprimé.")
    log("")
    log("  NOTE : Les logiciels existants (RealityScan, LichtFeld Studio, etc.)")
    log("  n'ont PAS été modifiés ou supprimés.")
    log("=" * 60)


if __name__ == "__main__":
    main()
