"""Create a portable source/data ZIP, excluding environments and generated runs."""
from pathlib import Path
import argparse
import zipfile

ROOT = Path(__file__).resolve().parent
EXCLUDED = {"outputs", ".venv", "venv", "__pycache__", ".git", ".pytest_cache"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "numerical-study-reproduction.zip")
    args = parser.parse_args()
    output = args.output.resolve()
    files = [p for p in sorted(ROOT.rglob("*")) if p.is_file()
             and not EXCLUDED.intersection(p.relative_to(ROOT).parts)
             and not any(part.startswith(".venv") for part in p.relative_to(ROOT).parts)
             and p.name not in {".DS_Store", "Thumbs.db"}
             and (p.suffix in {".py", ".md", ".txt", ".json", ".csv"} or p.name == ".gitignore")
             and p.suffix not in {".zip", ".pyc", ".log"}]
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            name = ROOT.name + "/" + path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 9, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    print(f"Packed {len(files)} files: {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
