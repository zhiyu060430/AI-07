"""Quiesced portable stack backups: PostgreSQL dump plus all mutable file volumes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from sqlalchemy.engine import make_url

ROOTS={'artifacts':Path('/app/artifacts'),'platform-state':Path('/app/.cache/platform'),
       'models':Path('/app/models'),'credentials':Path('/state'),'label-studio':Path('/label-data')}
DEST=Path('/backups')

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def database(command,*args):
    url=make_url(os.environ['AI07_DATABASE_URL'])
    subprocess.run([command,'-h',url.host,'-p',str(url.port or 5432),'-U',url.username,'-d',url.database,*args],
                   env={**os.environ,'PGPASSWORD':url.password},check=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['backup','restore','list'])
    parser.add_argument('--name');args=parser.parse_args();DEST.mkdir(exist_ok=True)
    if args.action=='list':
        for file in sorted(DEST.glob('*.tar')):print(file.name,file.stat().st_size)
        return
    name=args.name or datetime.now(timezone.utc).strftime('ai07-%Y%m%dT%H%M%SZ.tar')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*\.tar',name):raise ValueError('Invalid backup name')
    archive=DEST/name;checksum=DEST/(name+'.sha256')
    if args.action=='backup':
        if archive.exists() or checksum.exists():raise ValueError('Backup already exists')
        with tempfile.TemporaryDirectory(dir=DEST) as tmp:
            dump=Path(tmp)/'database.dump';database('pg_dump','-Fc','-f',str(dump))
            part=Path(tmp)/'backup.tar'
            with tarfile.open(part,'w') as tar:
                tar.add(dump,arcname='database.dump')
                for folder,path in ROOTS.items():
                    if path.exists():tar.add(path,arcname=folder)
            checksum.write_text(digest(part)+'\n');part.replace(archive)
        print(json.dumps({'backup':name,'bytes':archive.stat().st_size,'sha256':checksum.read_text().strip()}));return
    if not args.name:raise ValueError('Restore requires --name')
    if not checksum.is_file() or digest(archive)!=checksum.read_text().strip():raise ValueError('Backup checksum mismatch')
    # Fully validate and extract before replacing any running data.
    with tempfile.TemporaryDirectory(dir=DEST) as tmp:
        unpack=Path(tmp)
        with tarfile.open(archive) as tar:tar.extractall(unpack,filter='data')
        if not (unpack/'database.dump').is_file() or any(not (unpack/key).is_dir() for key in ROOTS):
            raise ValueError('Incomplete backup')
        # Keep the target database service password; restore all application accounts and other secrets.
        env_file=unpack/'credentials/compose.env'
        settings=dict(line.split('=',1) for line in env_file.read_text().splitlines() if '=' in line)
        settings['AI07_DB_PASSWORD']=make_url(os.environ['AI07_DATABASE_URL']).password
        # Network settings belong to the target machine, not to the archived machine.
        current=dict(line.split('=',1) for line in (ROOTS['credentials']/'compose.env').read_text().splitlines() if '=' in line)
        for key in ('AI07_WEB_PORT','AI07_API_PORT','AI07_LS_PORT','AI07_BIND_ADDRESS','AI07_ALLOWED_ORIGINS','AI07_LABEL_STUDIO_PUBLIC_URL','AI07_SECURE_COOKIE','AI07_GPU','AI07_REQUIRE_LOGIN'):
            if key in current:settings[key]=current[key]
        env_file.write_text(''.join(f'{k}={v}\n' for k,v in settings.items()))
        database('pg_restore','--clean','--if-exists','--no-owner','--exit-on-error','--single-transaction',str(unpack/'database.dump'))
        for folder,target in ROOTS.items():
            target.mkdir(parents=True,exist_ok=True)
            owner=(target.stat().st_uid,target.stat().st_gid)
            for child in target.iterdir():
                if child.is_dir() and not child.is_symlink():shutil.rmtree(child)
                else:child.unlink()
            shutil.copytree(unpack/folder,target,dirs_exist_ok=True,symlinks=True)
            # tar's data filter deliberately drops ownership. Label Studio runs as
            # an unprivileged user and must be able to update its restored SQLite DB.
            if folder=='label-studio' and hasattr(os,'chown'):
                os.chown(target,*owner)
                for child in target.rglob('*'):os.chown(child,*owner,follow_symlinks=False)
    print('Restored '+name+'; restart the stack with reloaded credentials.')

if __name__=='__main__':main()
