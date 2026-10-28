"""Run with stdlib Python in a temporary container; stdout is captured by launcher."""
import os
from pathlib import Path
import secrets

state=Path('/state');state.mkdir(exist_ok=True)
path=state/'compose.env'
values=dict(line.split('=',1) for line in path.read_text().splitlines() if '=' in line) if path.exists() else {}
for key in ('AI07_DB_PASSWORD','AI07_LS_PASSWORD','AI07_LS_TOKEN'):
    values.setdefault(key,secrets.token_hex(24))
defaults={'AI07_WEB_PORT':'3000','AI07_API_PORT':'8092','AI07_LS_PORT':'8082',
          'AI07_BIND_ADDRESS':'127.0.0.1','AI07_GPU':'0','AI07_SECURE_COOKIE':'0','AI07_REQUIRE_LOGIN':'0'}
for key,default in defaults.items():
    values[key]=os.getenv(key) or values.get(key,default)
for key in ('AI07_WEB_PORT','AI07_API_PORT','AI07_LS_PORT'):
    if not values[key].isdigit() or not 1<=int(values[key])<=65535:raise ValueError('Invalid port')
if values['AI07_GPU'] not in {'0','1'}:raise ValueError('Invalid GPU mode')
port=values['AI07_WEB_PORT']
values['AI07_ALLOWED_ORIGINS']=os.getenv('AI07_ALLOWED_ORIGINS') or values.get('AI07_ALLOWED_ORIGINS') or f'http://localhost:{port},http://127.0.0.1:{port}'
# Keep localhost origins consistent when the launcher explicitly changes port.
if os.getenv('AI07_WEB_PORT') and not os.getenv('AI07_ALLOWED_ORIGINS'):
    values['AI07_ALLOWED_ORIGINS']=f'http://localhost:{port},http://127.0.0.1:{port}'
values['AI07_LABEL_STUDIO_PUBLIC_URL']=os.getenv('AI07_LABEL_STUDIO_PUBLIC_URL') or f"http://localhost:{values['AI07_LS_PORT']}"
if any('\n' in v or '\r' in v for v in values.values()):raise ValueError('Invalid multiline setting')
os.umask(0o077)
path.write_text(''.join(f'{k}={v}\n' for k,v in values.items()))
key=Path('/platform/worker.key');key.parent.mkdir(parents=True,exist_ok=True)
if not key.exists():key.write_text(secrets.token_urlsafe(40))
for k,v in values.items():print(f'{k}={v}')
