"""研究条件から独立した、Mac・サーバーの保存先設定を読む（Python 3.10+）。"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ResearchPaths:
    project_root: Path
    data_root: Path
    output_root: Path
    cache_root: Path


def load_paths(config_path: str | Path, profile: str) -> ResearchPaths:
    """指定profileの絶対パスを返す。ディレクトリの作成や転送は行わない。"""
    with Path(config_path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    profiles = config.get("profiles") if isinstance(config, dict) else None
    if not isinstance(profiles, dict) or profile not in profiles:
        raise ValueError(f"保存先profileがありません: {profile}")
    selected = profiles[profile]
    if not isinstance(selected, dict):
        raise ValueError(f"profileはオブジェクトで指定してください: {profile}")
    paths = {}
    for key in ResearchPaths.__dataclass_fields__:
        value = selected.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"保存先を記入してください: {profile}.{key}")
        if "/absolute/path/to/" in value or "REPLACE_" in value:
            raise ValueError(f"保存先の例を実際のパスに変更してください: {profile}.{key}")
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError(f"絶対パスで指定してください: {profile}.{key}")
        paths[key] = path
    return ResearchPaths(**paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True, help="端末別保存先のJSON")
    parser.add_argument("--profile", required=True, help="mac / server等のprofile名")
    args = parser.parse_args()
    try:
        paths = load_paths(args.paths, args.profile)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({key: str(getattr(paths, key)) for key in paths.__dataclass_fields__},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
