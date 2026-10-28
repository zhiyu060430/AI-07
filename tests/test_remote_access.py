"""Private sharing must gate reads as well as writes and retain worker authentication."""
from pathlib import Path
import pytest
from test_platform import client,login


def test_architect_admin_and_product_member_permissions(client,monkeypatch):
    monkeypatch.setenv('AI07_MANAGED_ACCOUNTS','1')
    admin=login(client)
    assert admin['role']=='architect'
    assert client.get('/api/projects/ai07/users').status_code==200
    product=client.post('/api/projects/ai07/users',json={
        'username':'product_member','password':'test-password-123','role':'product'})
    assert product.status_code==200
    assert client.post('/api/projects',json={'name':'Admin project'}).status_code==200
    client.post('/api/logout')
    assert client.post('/api/login',json={'username':'product_member','password':'test-password-123'}).status_code==200
    assert client.get('/api/projects/ai07/users').status_code==403
    assert client.post('/api/projects/ai07/users',json={
        'username':'forbidden_member','password':'test-password-123','role':'architect'}).status_code==403
    assert client.post('/api/projects',json={'name':'Forbidden project'}).status_code==403
    assert client.post('/api/settings/label-studio',json={}).status_code==403


@pytest.mark.parametrize('route',[
    '/api/projects','/api/settings','/api/projects/ai07/samples',
    '/api/projects/ai07/events','/docs','/openapi.json','/metrics',
])
def test_private_mode_blocks_anonymous_data(client,monkeypatch,route):
    monkeypatch.setenv('AI07_REQUIRE_LOGIN','1')
    assert client.get('/api/auth').json()['require_login'] is True
    assert client.get(route).status_code==401
    assert client.get('/api/health').status_code==200


def test_private_mode_login_logout_and_internal_worker(client,monkeypatch):
    monkeypatch.setenv('AI07_REQUIRE_LOGIN','1')
    monkeypatch.setenv('AI07_MANAGED_ACCOUNTS','1')
    login(client)
    assert client.get('/api/projects/ai07/samples').status_code==200
    assert client.get('/docs').status_code==200
    client.post('/api/logout')
    assert client.get('/api/projects/ai07/samples').status_code==401
    assert client.post('/api/register',json={'username':'stranger','password':'test-password-123','role':'product'}).status_code==403
    assert client.post('/api/internal/claim').status_code==403
    key=(client.test_root/'.cache/platform/worker.key').read_text()
    assert client.post('/api/internal/claim',headers={'x-worker-key':key}).status_code==200


def test_private_mode_allows_only_configured_https_origin(client,monkeypatch):
    monkeypatch.setenv('AI07_REQUIRE_LOGIN','1')
    monkeypatch.setenv('AI07_ALLOWED_ORIGINS','https://team-demo.trycloudflare.com')
    login(client)
    client.post('/api/logout')
    body={'username':'manager','password':'test-only-long-password'}
    assert client.post('/api/login',json=body,headers={'Origin':'https://team-demo.trycloudflare.com'}).status_code==200
    assert client.post('/api/login',json=body,headers={'Origin':'https://stranger.example'}).status_code==403
