"""Local diagnostic provenance; file hashes are not loaded-bytecode identity."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from kaa.kaa_context import conf


def source_fingerprint(paths: list[Path], root: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        for path in sorted(set(paths)):
            digest.update(path.relative_to(root).as_posix().encode('utf-8'))
            digest.update(b'\0')
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest() if paths else None
    except (OSError, ValueError):
        return None


def local_code_provenance() -> dict:
    try:
        package_version = version('ksaa')
    except PackageNotFoundError:
        package_version = None
    kaa_root = Path(__file__).resolve().parents[5]
    play_root = Path(__file__).resolve().parents[1]
    paths = list(play_root.glob('*.py')) + list((play_root / 'planner').glob('*.py'))
    paths += [kaa_root / 'db' / name for name in (
        'card_growth.py', 'card_status_enchant.py', 'skill_card.py',
    )]
    return {
        'package_version': package_version,
        'source_sha256': source_fingerprint(paths, kaa_root),
        'source_files': len(paths),
        'source_scope': 'card_play_and_growth_python_files_on_disk',
        'sampling': 'first_input_per_battle',
    }


def configured_solution() -> dict:
    try:
        config = conf()
    except RuntimeError:
        return {'configured_solution_id': None, 'profile_config_sha256': None}
    return {
        'configured_solution_id': config.tasks.produce.selected_solution_id,
        'profile_config_sha256': hashlib.sha256(config.model_dump_json().encode('utf-8')).hexdigest(),
    }
