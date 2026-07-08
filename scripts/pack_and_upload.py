"""
Упаковать data/images_cps/ (финальные тайлы) в один tar-архив и залить на личный Google
Drive через rclone.

Зачем: тайлы регенерируются из data/cps/ и НЕ хранятся в git. Скрипт собирает их в один
архив (одним файлом быстрее и заливать, и распаковывать в Colab) и кладёт на Drive, откуда
train.ipynb забирает их в Colab по пути data/images_cps/ — пути совпадают с settings.py.

=====================================================================================
РАЗОВАЯ НАСТРОЙКА rclone (на каждой машине). Креды НЕ нужны: никаких API-ключей и
credentials.json — только одноразовый вход в свой Google-аккаунт через браузер.
=====================================================================================
  1. Установка rclone: https://rclone.org/install/
       Windows : winget install Rclone.Rclone   (или  scoop install rclone)
       Linux/Mac: curl https://rclone.org/install.sh | sudo bash
  2. Авторизация Drive:
       rclone config
         -> n                         # новый remote
         -> name>            gdrive
         -> Storage>         drive     # Google Drive
         -> client_id>       (Enter, пусто)
         -> client_secret>   (Enter, пусто)
         -> scope>           1          # полный доступ
         -> root_folder_id>  (Enter, пусто)
         -> service_account> (Enter, пусто, n)
         -> Edit advanced config> n
         -> Use auto config>  y         # откроется браузер — входите в Google-аккаунт
         -> Configure as Shared Drive> n
         -> y, затем q
  После этого rclone хранит OAuth-токен локально — повторная авторизация не нужна.

=====================================================================================
Использование:
=====================================================================================
  python scripts/pack_and_upload.py                       # упаковать + залить в gdrive:petroleum_data/
  python scripts/pack_and_upload.py --skip-upload          # только упаковать (dist/images_cps.tar)
  python scripts/pack_and_upload.py --remote gdrive --remote-path petroleum_data
  python scripts/pack_and_upload.py --gzip                 # сжать gzip (меньше размер, медленнее)
                                                            # ВАЖНО: тогда в Colab меняйте tar -xf на tar -xzf
"""

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from settings import settings  # noqa: E402


def count_files(source: Path) -> int:
    return sum(1 for p in source.rglob("*") if p.is_file())


def pack(source: Path, archive: Path, gzip: bool) -> None:
    mode = "w:gz" if gzip else "w"
    n = count_files(source)
    print(f"Упаковка {n} файлов из {source} -> {archive} ({'gzip' if gzip else 'без сжатия'}) ...")
    # arcname = 'images_cps' -> внутри архива корень images_cps/,
    # в Colab распаковываем в data/ и получаем data/images_cps/...
    with tarfile.open(archive, mode) as tar:
        tar.add(source, arcname=source.name)
    size_mb = archive.stat().st_size / 1048576
    print(f"Готово: {archive} ({size_mb:.1f} MiB)")


def upload(archive: Path, remote: str, remote_path: str) -> None:
    if not shutil.which("rclone"):
        print(
            "rclone не найден в PATH.\n"
            "Установите: https://rclone.org/install/  и выполните разовую настройку:\n"
            "  rclone config   (см. инструкции в шапке этого скрипта)"
        )
        sys.exit(1)

    dst = f"{remote}:{remote_path}/"
    print(f"Загрузка {archive.name} -> {dst} ...")
    subprocess.run(
        ["rclone", "copy", str(archive), dst, "--progress"],
        check=True,
    )
    print("Загрузка завершена.")
    print(
        "\nВ Colab (train.ipynb) архив будет доступен по пути:\n"
        f"  /content/drive/MyDrive/{remote_path}/{archive.name}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--source", default=settings.CPS_TILES_DIR,
                    help=f"папка-источник тайлов (по умолчанию {settings.CPS_TILES_DIR})")
    ap.add_argument("--out-dir", default="dist", help="куда положить архив (по умолчанию dist)")
    ap.add_argument("--name", default="images_cps.tar", help="имя архива")
    ap.add_argument("--gzip", action="store_true", help="сжать gzip (.tar.gz)")
    ap.add_argument("--skip-upload", action="store_true", help="только упаковать, не заливать")
    ap.add_argument("--remote", default="gdrive", help="имя remote в rclone (по умолчанию gdrive)")
    ap.add_argument("--remote-path", default="petroleum_data",
                    help="папка на Drive (по умолчанию petroleum_data)")
    args = ap.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        sys.exit(
            f"Источник не найден: {source}\n"
            "Сначала сгенерируйте тайлы:  python data/convert_cps_to_tiles.py"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / (args.name + (".gz" if args.gzip else ""))

    pack(source, archive, args.gzip)

    if args.skip_upload:
        print("--skip-upload: загрузка на Drive пропущена.")
        return

    upload(archive, args.remote, args.remote_path)


if __name__ == "__main__":
    main()
