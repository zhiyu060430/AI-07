"""Exercise real launchers against a fake Docker CLI without touching user containers."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def launcher(tmp_path):
    source=Path(__file__).resolve().parents[1]
    project=tmp_path/'source';project.mkdir();(project/'scripts').mkdir()
    for name in ('start.ps1','start.sh'):shutil.copy2(source/name,project/name)
    shutil.copy2(source/'scripts/portable-init.py',project/'scripts/portable-init.py')
    binary=tmp_path/'bin';binary.mkdir();log=tmp_path/'docker.jsonl'
    fake=binary/'fake_docker.py'
    fake.write_text('''import json,os,sys
from pathlib import Path
a=sys.argv[1:];log=Path(os.environ['STARTUP_TEST_LOG'])
with log.open('a') as f:f.write(json.dumps(a)+'\\n')
mode=os.environ.get('STARTUP_TEST_MODE','cached')
if a[:1]==['info']:print('29.3.1')
elif a[:2]==['image','inspect']:
    if mode!='cached' and a[2].endswith('-api:local'):sys.exit(1)
    print('sha256:test-local-image')
elif a[:1]==['pull']:raise AssertionError('Cached images must not be pulled')
elif a[:1]==['run']:
    if '-i' in a:
        sys.stdin.read()
        print('AI07_DB_PASSWORD=test-only-password\\nAI07_LS_PASSWORD=test-only-password\\nAI07_LS_TOKEN=test-only-token\\nAI07_WEB_PORT=3000\\nAI07_API_PORT=8092\\nAI07_LS_PORT=8082\\nAI07_GPU='+os.getenv('AI07_GPU','0'))
    else:print('ready')
elif a[:1]==['compose']:
    if 'build' in a:
        if mode=='cached':raise AssertionError('Cached startup must not build')
        previous=[json.loads(x) for x in log.read_text().splitlines()][:-1]
        if mode=='retry' and not any('build' in x for x in previous):
            print('failed to resolve source metadata: registry-1.docker.io: EOF',file=sys.stderr);sys.exit(1)
    if 'run' in a and '--no-build' in a:raise AssertionError('compose run has no --no-build flag')
else:raise AssertionError(a)
''',encoding='utf-8')
    if os.name=='nt':
        # cmd.exe needs CRLF and the Windows ANSI codepage here: the shim embeds the
        # repository path (which may contain non-ASCII characters), and a UTF-8 batch
        # file would be mis-decoded so the fake docker never runs.
        (binary/'docker.cmd').write_text(f'@echo off\r\n"{sys.executable}" "{fake}" %*\r\n',encoding='mbcs')
        shell=shutil.which('powershell')
        if not shell:pytest.skip('Windows PowerShell unavailable')
        command=[shell,'-NoProfile','-ExecutionPolicy','Bypass','-File',str(project/'start.ps1')]
    else:
        program=binary/'docker'
        program.write_text(f'#!{sys.executable}\nexec(compile(open({str(fake)!r}).read(),{str(fake)!r},"exec"))\n',encoding='utf-8')
        program.chmod(0o755)
        command=['sh',str(project/'start.sh')]
    env={k:v for k,v in os.environ.items() if not k.startswith('AI07_')}
    env.update(PATH=str(binary)+os.pathsep+env['PATH'],STARTUP_TEST_LOG=str(log))
    return project,command,env,log


@pytest.mark.parametrize('gpu',[False,True])
def test_prepared_startup_never_pulls_or_builds(launcher,gpu):
    project,command,env,log=launcher
    if gpu:command+=['-Gpu' if os.name=='nt' else '--gpu']
    result=subprocess.run(command,cwd=project,env=env,capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stdout+result.stderr
    calls=[json.loads(line) for line in log.read_text().splitlines()]
    assert not any('build' in a or a[:1]==['pull'] for a in calls)
    up=next(a for a in calls if a[:1]==['compose'] and 'up' in a)
    assert '--no-build' in up and up[up.index('--pull')+1]=='never'
    assert 'Skipping registry access' in result.stdout


def test_first_preparation_retries_registry_eof(launcher):
    project,command,env,log=launcher
    env['STARTUP_TEST_MODE']='retry'
    result=subprocess.run(command,cwd=project,env=env,capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr
    calls=[json.loads(line) for line in log.read_text().splitlines()]
    assert sum(a[:1]==['compose'] and 'build' in a and 'assets' in a for a in calls)==2
    assert 'retrying' in result.stdout.lower()+result.stderr.lower()
