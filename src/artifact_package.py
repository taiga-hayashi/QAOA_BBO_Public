"""明示した保存対象の梱包、SHA-256検証、rclone取得と新規領域への復元。"""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
import unicodedata
import zipfile

CHUNK = 1024 * 1024
EXCLUDED = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.ruff_cache',
            '.secrets', '.ssh', '.aws', 'cache', 'tmp', 'archive', '.DS_Store'}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError(f'不正な相対パス: {value!r}')
    p = PurePosixPath(value)
    if p.is_absolute() or any(x in {'', '.', '..'} for x in value.split('/')):
        raise ValueError(f'不正な相対パス: {value!r}')
    if any(ord(c) < 32 for c in value):
        raise ValueError('制御文字を含むパスは扱えません')
    return p.as_posix()


def identifier(value, max_length=128):
    if not isinstance(value, str) or not re.fullmatch(rf'[A-Za-z0-9][A-Za-z0-9._-]{{0,{max_length - 1}}}', value):
        raise ValueError(f'IDには英数字と . _ - を使ってください（{max_length}文字以内）')
    if value.startswith('REPLACE_'):
        raise ValueError('IDの例を実際の値に変更してください')
    return value


def digest_value(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', value):
        raise ValueError('取得前に信頼できる保存記録からSHA-256（64桁）を指定してください')
    return value.lower()


def excluded(parts):
    return any(p in EXCLUDED or p.startswith('.env') and p != '.env.example'
               or p.endswith(('.pem', '.key')) or p == 'credentials.json'
               or p.endswith('.local.json') for p in parts)


def checked_file(path):
    path = Path(path).expanduser()
    if not path.is_absolute() or '/absolute/path/to/' in str(path):
        raise ValueError('sourceには実際の絶対パスを指定してください')
    # 選択元の親を含め、リンクの向こう側を意図せず収集しない。
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError(f'シンボリックリンクは対象外です: {path}')
    return path


def stamp(path):
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f'通常ファイル以外は扱えません: {path}')
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def plan_package(spec):
    if spec.get('schema_version') != 1:
        raise ValueError('対応していない梱包仕様です')
    identifier(spec['package_id'], max_length=24)
    if spec.get('compute_status') not in {'partial', 'failed', 'complete'}:
        raise ValueError('compute_statusはpartial / failed / completeを明記してください')
    if (not isinstance(spec.get('run_ids'), list) or not spec['run_ids']
            or any(not isinstance(v, str) or not v or 'REPLACE_' in v for v in spec['run_ids'])
            or not isinstance(spec.get('study'), str) or not spec['study']
            or not isinstance(spec.get('code_version'), str) or not spec['code_version']
            or 'REPLACE_' in spec['study'] or 'REPLACE_' in spec['code_version']):
        raise ValueError('study・run_ids・code_versionを記入してください')
    if not isinstance(spec.get('sources'), list) or not spec['sources']:
        raise ValueError('sourcesで保存対象を明示してください')
    files, skipped, names = {}, [], set()
    for entry in spec['sources']:
        source = checked_file(entry['source'])
        target = safe_name(entry['target'])
        candidates = []
        if source.is_dir():
            for base, dirs, filenames in os.walk(source, followlinks=False):
                for d in dirs:
                    if excluded((d,)):
                        skipped.append(target + '/' + (Path(base) / d).relative_to(source).as_posix() + '/')
                dirs[:] = sorted(d for d in dirs if not excluded((d,)))
                for d in dirs:
                    if (Path(base) / d).is_symlink():
                        raise ValueError(f'シンボリックリンクを含みます: {Path(base) / d}')
                for name in sorted(filenames):
                    path = Path(base) / name
                    candidates.append((path, target + '/' + path.relative_to(source).as_posix()))
        elif source.is_file():
            candidates.append((source, target))
        else:
            raise ValueError(f'保存対象が見つかりません: {source}')
        for path, name in candidates:
            name = safe_name(name)
            if excluded(Path(name).parts) or excluded(path.relative_to(source.parent).parts):
                skipped.append(name)
                continue
            checked_file(path)
            key = unicodedata.normalize('NFC', name).casefold()
            if key in names or key == 'manifest.json':
                raise ValueError(f'保存先が重複しています: {name}')
            names.add(key)
            files[name] = {'source': path, 'stamp': stamp(path)}
    if not files:
        raise ValueError('保存対象が空です')
    for name in files:
        if any(unicodedata.normalize('NFC', str(p)).casefold() in names
               for p in PurePosixPath(name).parents if str(p) != '.'):
            raise ValueError(f'ファイルとディレクトリの保存先が衝突します: {name}')
    for required in spec.get('required_paths', []):
        required = safe_name(required)
        if not any(name == required or name.startswith(required + '/') for name in files):
            raise ValueError(f'必須ファイル・フォルダがありません: {required}')
    return files, skipped


def package_plan(spec):
    files, skipped = plan_package(spec)
    return {'package_id': spec['package_id'], 'files': sorted(files),
            'total_bytes': sum(v['stamp'][2] for v in files.values()), 'excluded_files': skipped}


def copy_hashed(source, destination):
    digest, size = hashlib.sha256(), 0
    for block in iter(lambda: source.read(CHUNK), b''):
        destination.write(block)
        digest.update(block)
        size += len(block)
    return size, digest.hexdigest()


def validate_zip(path, expected_sha256, destination=None):
    expected_sha256 = digest_value(expected_sha256)
    if sha256_file(path) != expected_sha256:
        raise ValueError('アーカイブのSHA-256が一致しません')
    with zipfile.ZipFile(path) as archive:
        entries, canonical = {}, set()
        for info in archive.infolist():
            name = safe_name(info.filename)
            key = unicodedata.normalize('NFC', name).casefold()
            mode = info.external_attr >> 16
            if key in canonical or info.is_dir() or info.flag_bits & 1:
                raise ValueError('重複・ディレクトリエントリ・暗号化ZIPは扱えません')
            if stat.S_IFMT(mode) not in {0, stat.S_IFREG}:
                raise ValueError('リンク等の特殊ファイルは展開しません')
            canonical.add(key)
            entries[name] = info
        if 'manifest.json' not in entries or entries['manifest.json'].file_size > 16 * CHUNK:
            raise ValueError('対応するmanifest.jsonがありません、または大きすぎます')
        manifest = json.loads(archive.read('manifest.json'))
        if manifest.get('schema_version') != 1 or not isinstance(manifest.get('files'), list):
            raise ValueError('対応していないmanifest形式です')
        expected = {}
        for row in manifest['files']:
            name = safe_name(row['path'])
            if name in expected or name == 'manifest.json':
                raise ValueError('manifestのパスが重複しています')
            if type(row['size']) is not int or row['size'] < 0:
                raise ValueError('manifestのサイズが不正です')
            expected[name] = {'size': row['size'], 'sha256': digest_value(row['sha256'])}
        if not expected or set(entries) != set(expected) | {'manifest.json'}:
            raise ValueError('ZIPとmanifestのファイル一覧が一致しません')
        for name in expected:
            if any(unicodedata.normalize('NFC', str(p)).casefold() in canonical
                   for p in PurePosixPath(name).parents if str(p) != '.'):
                raise ValueError('ファイルとディレクトリのパスが衝突しています')
        for name, row in expected.items():
            if entries[name].file_size != row['size']:
                raise ValueError(f'manifestのサイズが一致しません: {name}')
            with archive.open(name) as stream:
                size, digest = copy_hashed(stream, _NullWriter())
            if size != row['size'] or digest != row['sha256']:
                raise ValueError(f'内部ファイルのSHA-256が一致しません: {name}')
        if destination is not None:
            # 呼び出し元が専用の空ディレクトリを用意する。
            destination = Path(destination)
            if any(destination.iterdir()):
                raise ValueError('展開先は空である必要があります')
            for name in entries:
                out = destination / name
                out.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as stream, out.open('xb') as target:
                    copy_hashed(stream, target)
                mode = entries[name].external_attr >> 16
                out.chmod((mode & 0o777) or 0o600)
        return manifest


class _NullWriter:
    def write(self, block):
        pass


def pack(spec, output):
    if spec.get('snapshot_confirmed') is not True:
        raise ValueError('保存済みsnapshotを確認しsnapshot_confirmed=trueにしてください')
    files, skipped = plan_package(spec)
    output = Path(output).expanduser().absolute()
    if output.exists():
        raise FileExistsError('既存の保存先へ上書きしません')
    # 作成物自身や生成中のアーカイブを含めない。
    for entry in spec['sources']:
        source = Path(entry['source']).expanduser().absolute()
        if output == source or source in output.parents:
            raise ValueError('出力先を収集元の中に置かないでください')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.package-', dir=output.parent) as temporary:
        stage = Path(temporary)
        archive_path = stage / 'artifacts.zip'
        manifest = {'schema_version': 1, 'package_id': spec['package_id'],
                    'study': spec['study'], 'run_ids': spec['run_ids'],
                    'code_version': spec['code_version'], 'compute_status': spec['compute_status'],
                    'files': []}
        with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for name, row in sorted(files.items()):
                if stamp(row['source']) != row['stamp']:
                    raise ValueError('梱包中に元ファイルが変化しました')
                method = zipfile.ZIP_STORED if name.endswith(('.gz', '.zip', '.npz', '.png', '.tar')) else zipfile.ZIP_DEFLATED
                info = zipfile.ZipInfo(name)
                info.external_attr = (stat.S_IFREG | (row['source'].stat().st_mode & 0o777)) << 16
                info.compress_type = method
                with row['source'].open('rb') as source, archive.open(info, 'w', force_zip64=True) as target:
                    size, digest = copy_hashed(source, target)
                manifest['files'].append({'path': name, 'size': size, 'sha256': digest})
            archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        for row in files.values():
            if stamp(row['source']) != row['stamp']:
                raise ValueError('梱包中に元ファイルが変化しました。確定snapshotを使用してください')
        digest = sha256_file(archive_path)
        validate_zip(archive_path, digest)
        (stage / 'SHA256SUMS.txt').write_text(digest + '  artifacts.zip\n')
        report = {'package_id': spec['package_id'], 'archive_sha256': digest,
                  'files': len(files), 'package_validation_status': 'passed',
                  'storage_status': 'pending', 'excluded_files': skipped}
        (stage / 'package-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        (stage / 'README.md').write_text(
            '# 保存パッケージ\n\n' + f"- study: {spec['study']}\n- package_id: {spec['package_id']}\n"
            + f"- run_ids: {json.dumps(spec['run_ids'], ensure_ascii=False)}\n"
            + f"- code_version: {spec['code_version']}\n- compute_status: {spec['compute_status']}\n"
            + f'- artifacts.zip SHA-256: {digest}\n- パッケージ検証: passed\n'
            + '- Drive保存検証: pending（ローカル梱包のみ）\n'
            + '- 元データは保持。実験の検証・再現方法は同梱の記録を参照。\n')
        # 予約した新規ディレクトリだけを確定先にする。
        output.mkdir()
        try:
            for child in stage.iterdir():
                child.rename(output / child.name)
        except BaseException:
            shutil.rmtree(output)
            raise
    return report


class FetchError(ValueError):
    def __init__(self, message, completed):
        super().__init__(message)
        self.completed = list(completed)


def fetch(request, paths, archive_only=False):
    if request.get('schema_version') != 1:
        raise ValueError('対応していない取得指定です')
    records = request.get('packages')
    if not isinstance(records, list) or not records:
        raise ValueError('取得するpackagesを指定してください')
    plans, ids = [], set()
    for row in records:
        package_id = identifier(row['package_id'])
        if package_id.casefold() in ids:
            raise ValueError('package_idが重複しています')
        ids.add(package_id.casefold())
        digest = digest_value(row['sha256'])
        source = row['source']
        if source.get('kind') not in {'local', 'rclone'} or not source.get('path'):
            raise ValueError('source.kindはlocalまたはrcloneにしてください')
        if source['path'].startswith('-') or 'REPLACE_' in source['path']:
            raise ValueError('取得元の例を実際の値に変更してください')
        destination = paths.data_root / '_packages' / package_id
        if destination.exists() and not archive_only:
            raise FileExistsError(f'既存の展開先へ上書きしません: {destination}')
        plans.append((row, digest, source, destination))
    results = []
    for row, digest, source, destination in plans:
        try:
            legacy_cache = paths.cache_root / 'packages' / digest
            cache = legacy_cache if (legacy_cache / 'artifact.archive').is_file() else paths.cache_root / 'packages' / digest[:16]
            cache.mkdir(parents=True, exist_ok=True)
            archive_path = cache / 'artifact.archive'
            if archive_path.exists() and sha256_file(archive_path) != digest:
                raise ValueError('既存キャッシュが破損しています。確認して別保存先を指定してください')
            if not archive_path.exists():
                with tempfile.TemporaryDirectory(prefix='.fetch-', dir=cache) as temporary:
                    downloaded = Path(temporary) / 'artifact'
                    if source['kind'] == 'local':
                        shutil.copyfile(checked_file(source['path']), downloaded)
                    else:
                        # ディレクトリ誤指定による一括取得を防ぎ、ファイルだけを取得する。
                        metadata = json.loads(subprocess.check_output(
                            ['rclone', 'lsjson', '--stat', '--', source['path']], text=True))
                        if not isinstance(metadata, dict) or metadata.get('IsDir') is not False:
                            raise ValueError('rcloneの取得元は単一ファイルを指定してください')
                        # shellを使わず、新しい一時ファイルだけへ取得する。
                        subprocess.run(['rclone', 'copyto', '--', source['path'], str(downloaded)], check=True)
                    if not downloaded.is_file() or sha256_file(downloaded) != digest:
                        raise ValueError('取得したファイルのSHA-256が一致しません')
                    # 同じchecksumの同時取得でも既存キャッシュを上書きしない。
                    try:
                        os.link(downloaded, archive_path)
                    except FileExistsError:
                        if sha256_file(archive_path) != digest:
                            raise ValueError('同時取得のキャッシュが一致しません')
            result = {'package_id': row['package_id'], 'archive_path': str(archive_path),
                      'archive_sha256': digest, 'archive_verified': True,
                      'internal_manifest_verified': False, 'destination': None}
            if not archive_only:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix='.restore-', dir=destination.parent) as temporary:
                    stage = Path(temporary) / 'content'
                    stage.mkdir()
                    manifest = validate_zip(archive_path, digest, stage)
                    if manifest.get('package_id') != row['package_id']:
                        raise ValueError('取得対象とmanifestのpackage_idが一致しません')
                    destination.mkdir()
                    try:
                        for child in stage.iterdir():
                            child.rename(destination / child.name)
                    except BaseException:
                        shutil.rmtree(destination)
                        raise
                result.update(internal_manifest_verified=True, destination=str(destination),
                              payload_data_root=str(destination / 'data') if (destination / 'data').is_dir() else None)
            results.append(result)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
            raise FetchError(str(error), results) from error
    return results
