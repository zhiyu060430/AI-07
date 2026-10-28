"""Container-native plans must preserve stages and enforce managed account roles."""
import sys
import pytest
from platform_api.worker import native_command
from test_platform import client
from test_platform import login,payload


@pytest.mark.parametrize('service,module,stage',[
    ('physics','fire_modeling.workflow','simulate-web'),
    ('blender','fire_modeling.workflow','video'),
    ('enhance','fire_modeling.workflow','synthesis-enhance'),
    ('gpu','fire_modeling','distribution'),
])
def test_native_stage_preserves_arguments(service,module,stage):
    args=['docker','compose','run','--rm','--no-deps','--name','job',service,stage,'--root','/app/artifacts/job']
    assert native_command(args,'job')==[sys.executable,'-m',module,stage,'--root','/app/artifacts/job']


def test_native_plan_rejects_unknown_service():
    with pytest.raises(ValueError,match='Unsupported'):
        native_command(['docker','compose','run','--no-deps','--name','job','shell','bad'],'job')


def test_managed_accounts_first_admin_and_no_self_elevation(client,monkeypatch):
    monkeypatch.setenv('AI07_MANAGED_ACCOUNTS','1')
    assert client.get('/api/auth').json()['setup_required']
    first=client.post('/api/register',json={'username':'owner','password':'long-password-123','role':'tester'})
    assert first.status_code==200 and first.json()['role']=='architect'
    client.post('/api/logout')
    assert client.get('/api/auth').json()['registration_open'] is False
    second=client.post('/api/register',json={'username':'second','password':'long-password-123','role':'product'})
    assert second.status_code==403
    assert client.post('/api/login',json={'username':'owner','password':'long-password-123'}).status_code==200
    added=client.post('/api/projects/ai07/users',json={'username':'member','password':'long-password-456','role':'tester'})
    assert added.status_code==200 and added.json()['role']=='tester'


def test_cpu_deployment_rejects_unprepared_gpu_job(client,monkeypatch):
    monkeypatch.setenv('AI07_GPU_ENABLED','0')
    login(client)
    data=payload();data.update(route='physics',enhance=True)
    result=client.post('/api/projects/ai07/jobs',json=data)
    assert result.status_code==400 and 'GPU' in result.json()['detail']
