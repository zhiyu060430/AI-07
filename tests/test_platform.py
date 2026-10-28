"""Application contracts: authorization, isolation, review gates, real DVC and worker leases."""
import copy
import io
import json
from pathlib import Path
import shutil
import zipfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine,select
from sqlalchemy.orm import sessionmaker
from platform_api import app as web, services, integrations
from platform_api.db import Base,User,Member,Run,Job,Version
from fire_modeling.config import SceneConfig
from fire_modeling.pipeline import run_pipeline

@pytest.fixture
def client(tmp_path,monkeypatch):
    root=tmp_path/'project'; root.mkdir(); (root/'configs').mkdir(); state=root/'.cache/platform';state.mkdir(parents=True)
    source=Path(__file__).resolve().parents[1]
    for name in ['room.json','outdoor.json','label_studio.xml']:shutil.copy(source/'configs'/name,root/'configs'/name)
    (root/'SRS.md').write_text('Test requirement',encoding='utf-8')
    config=SceneConfig(width=64,height=64,plume_steps=2,particles_per_step=1)
    run_pipeline(config,root/'artifacts/demo/room_demo',count=2,enhancer='none')
    engine=create_engine('sqlite:///'+str(root/'test.db'),connect_args={'check_same_thread':False})
    factory=sessionmaker(engine,expire_on_commit=False)
    for module in [web,services,integrations]:
        monkeypatch.setattr(module,'ROOT',root);monkeypatch.setattr(module,'STATE',state)
    for module in [web,services]:monkeypatch.setattr(module,'Session',factory)
    for module in [web,services]:monkeypatch.setattr(module,'engine',engine)
    with TestClient(web.app) as c:
        c.test_root=root;c.factory=factory
        yield c
    engine.dispose()

def login(client):
    response=client.post('/api/setup',json={'username':'manager','password':'test-only-long-password'})
    assert response.status_code==200
    return response.json()


def test_video_scene_revision_auth_and_resource_limits(client):
    from fire_modeling.editor_contract import upstream
    raw={'scenario':'房间火灾','origin_fuel':'木材','placements':upstream().catalog_extras()['defaults']['房间火灾'],'viewpoints':['侧视']}
    assert client.post('/api/projects/ai07/simulation-videos',json={}).status_code==401
    login(client)
    saved=client.put('/api/projects/ai07/scenes/room',json={'expected_revision':0,'editor_scene':raw}).json()
    body={'scene_id':'room','scene_revision':saved['config']['editor_scene']['revision'],'parameters':{'duration_s':4,'width':256,'height':256},'idempotency_key':'video-contract'}
    response=client.post('/api/projects/ai07/simulation-videos',json=body)
    assert response.status_code==200 and response.json()['kind']=='video'
    assert client.get('/api/simulation-videos/'+response.json()['id']).status_code==409
    assert client.post('/api/projects/ai07/simulation-videos',json={**body,'scene_revision':0}).status_code==409
    assert client.post('/api/projects/ai07/simulation-videos',json={**body,'parameters':{'duration_s':30}}).status_code==400
    data={'source_simulation_id':response.json()['id'],'count':2,'idempotency_key':'bad-source-contract'}
    assert client.post('/api/projects/ai07/jobs',json=data).status_code==400


def test_original_editor_submission_preserves_controls_and_scope(client):
    from fire_modeling.editor_contract import upstream
    raw={'scenario':'房间火灾','origin_fuel':'木材','placements':upstream().catalog_extras()['defaults']['房间火灾'],'viewpoints':['侧视']}
    assert client.post('/api/projects/ai07/scene-editor/generate',json={}).status_code==401
    login(client);client.put('/api/projects/ai07/scenes/room',json={'expected_revision':0,'editor_scene':raw})
    body={'scene_id':'room','scene_revision':1,'generation':{'frame_count':4,'humidity':62,'weathers':['雨','雪'],'augmentations':['噪声'],'count':2,'wind_step':.4},'idempotency_key':'original-port-001'}
    r=client.post('/api/projects/ai07/scene-editor/generate',json=body)
    assert r.status_code==200 and r.json()['kind']=='editor'
    assert r.json()['payload']['generation']['humidity']==62
    j=r.json()['id'];list_url='/api/projects/ai07/scene-editor/compat/jobs'
    assert client.get(list_url+'?scene=room').json()['jobs'][0]['job_id']==j
    assert client.get(list_url+'/'+j+'/stages').status_code==200
    assert client.get('/api/projects/other/scene-editor/compat/jobs/'+j).status_code==403
    bad={**body,'generation':{'engine':'fds','weathers':['雪']}}
    assert client.post('/api/projects/ai07/scene-editor/generate',json=bad).status_code==400
    assert client.post('/api/projects/ai07/scene-editor/generate',json={**body,'scene_revision':0}).status_code==409


def test_scene_editor_save_preview_revision_and_auth(client):
    from fire_modeling.editor_contract import upstream
    p=upstream();raw={'scenario':'房间火灾','origin_fuel':'木材','placements':p.catalog_extras()['defaults']['房间火灾'],'viewpoints':['侧视']}
    endpoint='/api/projects/ai07/scenes/room'
    assert client.get('/api/projects/ai07/scene-editor/catalog').status_code==200
    assert client.put(endpoint,json={'editor_scene':raw,'expected_revision':0}).status_code==401
    login(client)
    response=client.put(endpoint,json={'editor_scene':raw,'expected_revision':0})
    assert response.status_code==200,response.text
    config=response.json()['config'];assert config['editor_scene']['revision']==1
    assert client.put(endpoint,json={'editor_scene':raw,'expected_revision':0}).status_code==409
    preview=client.post('/api/projects/ai07/scene-editor/preview',json=config['editor_scene'])
    assert preview.status_code==200 and '侧视' in preview.json()['cameras']
    assert client.put('/api/projects/other/scenes/room',json={'editor_scene':raw,'expected_revision':1}).status_code==404
    queued=client.post('/api/projects/ai07/jobs',json={'config':config,'route':'physics','count':1,'enhance':False,'idempotency_key':'editor-scene-job'})
    assert queued.status_code==200,queued.text
    assert queued.json()['payload']['config']['editor_scene']['revision']==1


def test_scene_editor_model_scope_and_malformed_upload(client):
    from platform_api.scene_editor import assert_assets,asset_file
    login(client)
    response=client.post('/api/projects/ai07/scene-assets',files={'file':('fake.glb',b'invalid')})
    assert response.status_code==400
    assert client.get('/api/projects/ai07/scene-assets/upload/0123456789abcdef.glb').status_code==400
    with pytest.raises(ValueError):assert_assets(client.test_root,'ai07',{'placements':[{'asset_id':'upload:0123456789abcdef'}]})
    with pytest.raises(ValueError):asset_file(client.test_root,'ai07','builtin','../secret.glb')


def test_scene_editor_uploaded_glb_is_project_scoped(client):
    import struct
    login(client)
    document={'asset':{'version':'2.0'},'accessors':[{'count':3,'min':[0,0,0],'max':[1,1,1]}],
              'meshes':[{'primitives':[{'attributes':{'POSITION':0}}]}]}
    def encode(doc):
        raw=json.dumps(doc).encode();raw+=b' '*((-len(raw))%4)
        return struct.pack('<4sII',b'glTF',2,20+len(raw))+struct.pack('<II',len(raw),0x4e4f534a)+raw
    response=client.post('/api/projects/ai07/scene-assets',files={'file':('test.glb',encode(document))})
    assert response.status_code==200,response.text
    asset=response.json();assert client.get(asset['url']).content==encode(document)
    from platform_api.scene_editor import assert_assets
    with pytest.raises(ValueError):assert_assets(client.test_root,'other',{'placements':[asset]})
    document['buffers']=[{'uri':'https://example.org/private.bin','byteLength':12}]
    assert client.post('/api/projects/ai07/scene-assets',files={'file':('external.glb',encode(document))}).status_code==400

def sample(client):return client.get('/api/projects/ai07/samples').json()[1]

def payload():return {'name':'Preview contract','config':SceneConfig(width=64,height=64).to_dict(),'route':'preview','count':2,'enhance':False,'idempotency_key':'contract-job-one'}

def test_guest_catalog_has_real_file_backing(client):
    assert client.get('/api/auth').json()['setup_required']
    overview=client.get('/api/projects/ai07/overview').json()
    assert overview['samples']==2 and overview['approved']==0 and overview['real']==0
    s=sample(client); assert client.get(s['url']).content.startswith(b'\x89PNG')
    assert client.get(s['mask_url']).status_code==200
    assert client.post('/api/projects/ai07/jobs',json=payload()).status_code==401

def test_setup_and_session_lifecycle(client):
    login(client)
    assert client.post('/api/setup',json={}).status_code==409
    assert client.get('/api/auth').json()['user']['role']=='architect'
    assert client.post('/api/logout').status_code==200
    assert client.get('/api/auth').json()['user'] is None
    assert client.post('/api/login',json={'username':'manager','password':'wrong-password'}).status_code==401
    assert client.post('/api/login',json={'username':'manager','password':'test-only-long-password'}).status_code==200

def test_setup_validation_and_cross_origin_block(client):
    assert client.post('/api/setup',json={'username':'bad/name','password':'123'}).status_code==400
    assert client.post('/api/setup',json={'username':'manager','password':'123'}).status_code==400
    assert client.post('/api/setup',headers={'Origin':'https://evil.example'},json={}).status_code==403

def test_roles_and_project_isolation(client):
    login(client)
    p=client.post('/api/projects',json={'name':'Private research'}).json()['id']
    assert client.post('/api/projects/ai07/users',json={'username':'developer','password':'developer-test-only','role':'developer'}).status_code==200
    assert len(client.get('/api/projects/ai07/users').json())==2
    assert client.post('/api/projects/ai07/users',json={'username':'developer','password':'developer-test-only','role':'developer'}).status_code==400
    client.post('/api/logout');client.post('/api/login',json={'username':'developer','password':'developer-test-only'})
    assert client.get(f'/api/projects/{p}/overview').status_code==403
    assert client.post('/api/projects',json={'name':'forbidden'}).status_code==403
    s=sample(client)
    assert client.post(f'/api/runs/{s["run_id"]}/review',json={'sample_id':s['sample_id'],'decision':'approved'}).status_code==403
    assert client.post('/api/settings/label-studio',json={}).status_code==403

def test_scene_validation_and_persistence(client):
    login(client);assert len(client.get('/api/projects/ai07/scenes').json())==2
    assert client.post('/api/projects/ai07/scenes',json={'name':'new','config':SceneConfig().to_dict()}).status_code==200
    invalid=SceneConfig().to_dict();invalid['wind_m_s']=1
    assert client.post('/api/projects/ai07/scenes',json={'config':invalid}).status_code==400

def test_task_idempotency_cancel_and_retry(client):
    login(client)
    a=client.post('/api/projects/ai07/jobs',json=payload()).json()
    b=client.post('/api/projects/ai07/jobs',json=payload()).json();assert a['id']==b['id']
    assert len(client.get('/api/projects/ai07/jobs').json())==1
    assert client.post(f'/api/jobs/{a["id"]}/cancel').json()['status']=='cancelled'
    retry=client.post(f'/api/jobs/{a["id"]}/retry').json();assert retry['id']!=a['id']
    assert client.post(f'/api/jobs/{retry["id"]}/retry').status_code==400
    invalid={**payload(),'count':0};assert client.post('/api/projects/ai07/jobs',json=invalid).status_code==400

def test_review_rejects_invalid_boxes_and_changed_image(client):
    login(client);s=sample(client);route=f'/api/runs/{s["run_id"]}/review'
    assert client.post(route,json={'sample_id':s['sample_id'],'decision':'approved','boxes':[[-1,0,3,3]]}).status_code==400
    assert client.post(route,json={'sample_id':s['sample_id'],'decision':'approved','boxes':s['boxes']}).status_code==200
    current=sample(client);assert current['review_status']=='approved'
    assert client.post(route,json={'sample_id':s['sample_id'],'decision':'rejected','boxes':[]}).status_code==200
    assert sample(client)['review_status']=='rejected'
    assert client.post(f'/api/runs/{s["run_id"]}/quality').json()['count']==2

def test_publish_cannot_bypass_review_or_overwrite(client):
    login(client);s=sample(client)
    assert client.post('/api/projects/ai07/versions',json={'run_id':s['run_id'],'name':'v1'}).status_code==400
    result=client.post('/api/projects/ai07/versions',json={'run_id':s['run_id'],'name':'v1','research_snapshot':True})
    assert result.status_code==409 and 'approved' in result.json()['detail']
    assert client.get('/api/projects/ai07/versions').json()[0]['status']=='failed'
    assert client.post('/api/projects/ai07/versions',json={'run_id':s['run_id'],'name':'v1','research_snapshot':True}).status_code==400

@pytest.mark.skipif(not shutil.which('git'),reason='Git required for DVC integration')
def test_real_dvc_snapshot_and_download(client):
    login(client);s=sample(client)
    client.post(f'/api/runs/{s["run_id"]}/review',json={'sample_id':s['sample_id'],'decision':'approved','boxes':s['boxes']})
    response=client.post('/api/projects/ai07/versions',json={'run_id':s['run_id'],'name':'research-v1','research_snapshot':True})
    assert response.status_code==200,response.text
    result=response.json();assert len(result['result']['commit'])==40
    archive=client.get(f'/api/versions/{result["id"]}/download')
    with zipfile.ZipFile(io.BytesIO(archive.content)) as z:assert 'coco.json' in z.namelist() and 'data.yaml' in z.namelist()
    assert (client.test_root/'.cache/platform/dvc-repository/datasets'/f'{result["id"]}.dvc').is_file()

def test_worker_auth_lease_and_no_fake_success(client):
    login(client);job=client.post('/api/projects/ai07/jobs',json=payload()).json()
    assert client.post('/api/internal/claim').status_code==403
    headers={'x-worker-key':(client.test_root/'.cache/platform/worker.key').read_text()}
    claimed=client.post('/api/internal/claim',headers=headers).json();assert claimed['id']==job['id']
    assert client.post('/api/internal/claim',headers=headers).json() is None
    endpoint=f'/api/internal/jobs/{job["id"]}'
    assert client.post(endpoint,headers=headers,json={'lease':'stale','status':'succeeded'}).status_code==409
    assert client.post(endpoint,headers=headers,json={'lease':claimed['lease'],'status':'succeeded'}).status_code==400
    failed=client.post(endpoint,headers=headers,json={'lease':claimed['lease'],'status':'failed','log':'failure evidence','result':{'error':'test engine unavailable'}})
    assert failed.status_code==200
    assert client.get('/api/projects/ai07/jobs').json()[0]['status']=='failed'
    assert client.post(endpoint,headers=headers,json={'lease':claimed['lease']}).status_code==409

def test_worker_success_registers_actual_manifest(client):
    login(client);job=client.post('/api/projects/ai07/jobs',json=payload()).json()
    headers={'x-worker-key':(client.test_root/'.cache/platform/worker.key').read_text()}
    claimed=client.post('/api/internal/claim',headers=headers).json()
    target=client.test_root/f'artifacts/platform/jobs/{job["id"]}/render'
    shutil.copytree(client.test_root/'artifacts/demo/room_demo',target)
    assert client.post(f'/api/internal/jobs/{job["id"]}',headers=headers,json={'lease':claimed['lease'],'status':'succeeded','result':{'elapsed_s':1}}).status_code==200
    assert len(client.get('/api/projects/ai07/overview').json()['runs'])==2

def test_label_settings_no_secret_leak_or_public_ssrf(client):
    login(client)
    assert client.post('/api/settings/label-studio',json={'url':'http://evil.example','token':'secret'}).status_code==400
    assert client.post('/api/settings/label-studio',json={'url':'http://localhost:8080','token':'test-secret-token'}).status_code==200
    assert 'test-secret-token' not in client.get('/api/settings').text
    s=sample(client)
    assert client.post(f'/api/runs/{s["run_id"]}/label-studio/pull').status_code==400

def test_physical_and_detection_metrics_are_real_computations(client):
    login(client)
    ref={'quantity':'temperature','unit':'C','time_s':[0,1,2],'values':[20,40,60]}
    prediction={**ref,'values':[20,42,60]}
    r=client.post('/api/projects/ai07/experiments',json={'kind':'physical','reference':ref,'prediction':prediction})
    assert r.status_code==200 and r.json()['result']['normalized_absolute_error']==pytest.approx(2/120)
    truth={'images':[{'id':1,'width':64,'height':64}],'annotations':[{'image_id':1,'category_id':1,'bbox':[1,1,10,10]}]}
    pred=[{'image_id':1,'category_id':1,'bbox':[1,1,10,10],'score':.9}]
    r=client.post('/api/projects/ai07/experiments',json={'kind':'detection','truth':truth,'predictions':pred})
    assert r.status_code==200 and r.json()['result']['AP50']==1
    assert len(client.get('/api/projects/ai07/experiments').json())==2

def test_fid_needs_independent_real_import(client):
    login(client);s=sample(client)
    assert client.post('/api/projects/ai07/experiments',json={'kind':'fid','real_run':s['run_id'],'synthetic_run':s['run_id']}).status_code==400


def test_native_label_bridge_idempotent_revisions_and_conflict(client,monkeypatch):
    from platform_api import label_bridge
    from platform_api.db import Link
    from fire_modeling.io import read_json
    login(client);s=sample(client);tasks=[{'id':7,'data':{'sample_id':s['sample_id']},'annotations':[]}];calls=[]
    def request(config,method,path,**kwargs):
        calls.append((method,path))
        if method=='GET':return copy.deepcopy(tasks)
        if method=='POST':
            a={**kwargs['json'],'id':8};tasks[0]['annotations']=[a];return a
        if method=='PATCH':tasks[0]['annotations'][0].update(kwargs['json']);return tasks[0]['annotations'][0]
        raise AssertionError(path)
    monkeypatch.setattr(integrations,'label_request',request)
    monkeypatch.setattr(integrations,'label_sync',lambda db,run:{'project_id':1})
    with client.factory.begin() as db:db.add(Link(run_id=s['run_id'],value={'project_id':1,'url':'http://localhost:8082/projects/1/data'}))
    path=f'/api/runs/{s["run_id"]}'
    saved=client.post(path+'/review',json={'sample_id':s['sample_id'],'decision':'approved','boxes':s['boxes']})
    assert saved.status_code==200 and not calls
    assert client.get(path+'/label-studio').json()['pending_sync']==1
    assert client.post(path+'/label-studio/push').json()['synced']==1
    assert any('download_all_tasks=true' in path for method,path in calls if method=='GET')
    assert client.post(path+'/label-studio/push').json()['synced']==0
    assert len([x for x in calls if x[0]=='POST'])==1
    client.post(path+'/review',json={'sample_id':s['sample_id'],'decision':'rejected','boxes':[]})
    assert client.post(path+'/label-studio/push').json()['synced']==1
    assert tasks[0]['annotations'][0]['was_cancelled']
    client.post(path+'/review',json={'sample_id':s['sample_id'],'decision':'approved','boxes':s['boxes']})
    tasks[0]['annotations'][0]['result']=[{'edited':'externally'}]
    assert client.post(path+'/label-studio/push').status_code==400
    assert client.get(path+'/label-studio').json()['pending_sync']==1
    assert client.get('/api/projects/ai07/workflow').json()['runs']


def test_reviewed_distribution_and_detector_endpoints_need_evidence(client):
    login(client);s=sample(client)
    assert client.get('/api/projects/ai07/evaluations/capabilities').status_code==200
    assert client.post('/api/projects/ai07/evaluations/detector-comparisons',json={}).status_code in {403,404}
    with client.factory.begin() as db:
        run=db.get(Run,s['run_id']);run.route='real'
    assert client.post('/api/projects/ai07/evaluations/distribution',json={'real_run':s['run_id'],'synthetic_run':s['run_id']}).status_code==400


def test_distribution_freezes_only_reviewed_images_and_model_job_is_versioned(client):
    from PIL import Image
    from fire_modeling.io import write_json,sha256,read_json
    from fire_modeling.dataset import review
    from fire_modeling.evaluation_data import prepare_detector_pair
    from platform_api import worker
    login(client)
    for name,source in [('test-real','real'),('test-synthetic','fds_openvdb_volume')]:
        root=client.test_root/'artifacts'/name;rows=[]
        for i in range(3):
            image=root/f'images/{i}.png';image.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(64,64),(10+i*30,100 if source=='real' else 150,7)).save(image)
            rows.append({'sample_id':str(i),'source':source,'image':f'images/{i}.png','sha256':sha256(image),'width':64,'height':64,'boxes':[[1,2,10,12]],'parent_id':None,'group_id':name+str(i)})
        write_json(root/'manifest.json',{'status':'success_candidates_only','samples':rows})
        for i in range(2):review(root,str(i),'test-fixture-reviewer')
        with client.factory.begin() as db:db.add(Run(id=name,project_id='ai07',name=name,relative_path='artifacts/'+name,route='real' if source=='real' else 'physics'))
    submitted=client.post('/api/projects/ai07/evaluations/distribution',json={'real_run':'test-real','synthetic_run':'test-synthetic','selection':'raw','metric':'color-js'})
    assert submitted.status_code==200,submitted.text
    job=submitted.json();assert job['payload']['evaluation_snapshot']['counts']=={'real':2,'synthetic':2}
    snapshot=client.test_root/job['payload']['synthetic_path'];assert len(list(snapshot.glob('*.png')))==2
    assert '2' not in {s['sample_id'] for s in read_json(snapshot.parent/'snapshot.json')['samples']}
    monkey_root=worker.ROOT;worker.ROOT=client.test_root
    try:assert 'color-js' in worker.command_plan(job)[0][1]
    finally:worker.ROOT=monkey_root
    # Independent version fixtures, not acceptance data or dashboard results.
    from test_evaluation_workflow import version
    ids=['a'*24,'b'*24]
    for id,source in zip(ids,['real','diffusion']):
        p=client.test_root/'.cache/platform/dvc-repository/datasets'/id;version(p,source)
        with client.factory.begin() as db:db.add(Version(id=id,project_id='ai07',name=id,run_id='test',status='published'))
    weights=client.test_root/'models/yolo11n.pt';weights.parent.mkdir();weights.write_bytes(b'test-placeholder-weights-not-an-actual-model')
    response=client.post('/api/projects/ai07/evaluations/detector-comparisons',json={'real_version':ids[0],'synthetic_version':ids[1],'epochs':1,'imgsz':64})
    assert response.status_code==200,response.text
    assert response.json()['kind']=='detector-pair'
    worker.ROOT=client.test_root
    try:assert worker.command_plan(response.json())[0][0]=='detector-pair'
    finally:worker.ROOT=monkey_root

def test_import_blocks_zip_traversal_and_registers_real_source(client):
    login(client);s=sample(client)
    invalid=io.BytesIO()
    with zipfile.ZipFile(invalid,'w') as z:z.writestr('../escape.txt','bad')
    assert client.post('/api/projects/ai07/import-coco',files={'file':('bad.zip',invalid.getvalue())},data={'source_url':'https://example.org/data','license_note':'test license'}).status_code==400
    content=io.BytesIO();image=client.get(s['url']).content
    coco={'images':[{'id':1,'file_name':'a.png','width':64,'height':64}],'annotations':[],'categories':[{'id':1,'name':'fire'}]}
    with zipfile.ZipFile(content,'w') as z:z.writestr('images/a.png',image);z.writestr('coco.json',json.dumps(coco))
    r=client.post('/api/projects/ai07/import-coco',files={'file':('real.zip',content.getvalue())},data={'source_url':'https://example.org/data','license_note':'test-only authorized example'})
    assert r.status_code==200 and r.json()['route']=='real'
    assert client.get('/api/projects/ai07/overview').json()['real']==1

def test_import_coco_accepts_link_or_upload_but_not_neither(client):
    login(client)
    from platform_api.remote_import import assert_public_url
    import pytest as _pytest
    assert_public_url('http://198.18.3.191/x.zip')  # local VPN fake-IP range is forwarded, not intranet
    with _pytest.raises(ValueError):
        assert_public_url('http://192.168.1.10/x.zip')
    r=client.post('/api/projects/ai07/import-coco',data={'source_url':'','license_note':''})
    assert r.status_code==400 and '二选一' in r.json()['detail']
    r=client.post('/api/projects/ai07/import-coco',data={'source_url':'http://127.0.0.1:9/x.zip'})
    assert r.status_code==400 and '内网' in r.json()['detail']

def test_import_coco_from_link_samples_fire_first_and_rewrites_coco(client,monkeypatch):
    login(client)
    from PIL import Image
    import platform_api.remote_import as remote
    buf=io.BytesIO();Image.new('RGB',(64,64),(200,40,10)).save(buf,'PNG');png=buf.getvalue()
    coco={'images':[{'id':1,'file_name':'fire1.png','width':64,'height':64},{'id':2,'file_name':'fire2.png','width':64,'height':64},{'id':3,'file_name':'neg.png','width':64,'height':64}],
          'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[1,1,10,10],'iscrowd':0},{'id':2,'image_id':2,'category_id':1,'bbox':[1,1,10,10],'iscrowd':0}],
          'categories':[{'id':1,'name':'fire'}]}
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('data/coco.json',json.dumps(coco))
        z.writestr('data/images/fire1.png',png);z.writestr('data/images/fire2.png',png);z.writestr('data/images/neg.png',png)
    def fake_fetch(url,workdir):
        path=workdir/'remote.zip';path.write_bytes(archive.getvalue());return url,path,'ds.zip'
    monkeypatch.setattr(remote,'fetch_remote_archive',fake_fetch)
    monkeypatch.setattr(remote,'MAX_FILES',2)
    r=client.post('/api/projects/ai07/import-coco',data={'source_url':'https://acquire.cqu.edu.au/articles/dataset/example/28747046','license_note':''})
    assert r.status_code==200,r.text
    assert r.json()['route']=='real' and r.json()['name']=='ds.zip'
    assert r.json()['sampling']=={'mode':'coco','kept':2,'total':3,'skipped':1,'missing':0}
    samples=[s for s in client.get('/api/projects/ai07/samples').json() if s['run_id']==r.json()['id']]
    assert len(samples)==2 and all(s['review_status']=='pending' for s in samples)

def test_import_coco_from_link_supports_yolo_archives(client,monkeypatch):
    login(client)
    from PIL import Image
    import platform_api.remote_import as remote
    buf=io.BytesIO();Image.new('RGB',(96,64),(200,40,10)).save(buf,'PNG');fire_png=buf.getvalue()
    buf=io.BytesIO();Image.new('RGB',(64,64),(40,60,80)).save(buf,'PNG');neg_png=buf.getvalue()
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('data.yaml','train: train/images\nnames:\n  - fire\n  - smoke\n')
        z.writestr('train/images/f1.png',fire_png);z.writestr('train/labels/f1.txt','0 0.5 0.5 0.4 0.6\n1 0.1 0.1 0.1 0.1\n')
        z.writestr('train/images/f2.png',fire_png);z.writestr('train/labels/f2.txt','0 0.25 0.25 0.2 0.2\n')
        z.writestr('train/images/n1.png',neg_png);z.writestr('train/labels/n1.txt','1 0.5 0.5 0.2 0.2\n')
        z.writestr('train/images/x1.png',neg_png)  # no label file: ignored like the CLI converter
    def fake_fetch(url,workdir):
        path=workdir/'remote.zip';path.write_bytes(archive.getvalue());return url,path,'yolo.zip'
    monkeypatch.setattr(remote,'fetch_remote_archive',fake_fetch)
    r=client.post('/api/projects/ai07/import-coco',data={'source_url':'https://example.org/ds','license_note':''})
    assert r.status_code==200,r.text
    assert r.json()['sampling']=={'mode':'yolo','kept':3,'total':3,'skipped':0,'missing':0}
    samples=[s for s in client.get('/api/projects/ai07/samples').json() if s['run_id']==r.json()['id']]
    fire=[s for s in samples if s['sample_id'].endswith('_1')]
    assert len(fire)==1 and len(fire[0]['boxes'])==1
    boxes_by_id={s['sample_id'].rsplit('_',1)[-1]:len(s['boxes']) for s in samples}
    assert boxes_by_id=={'1':1,'2':1,'3':0}

def test_evidence_is_honest_and_document_allowlist(client):
    response=client.get('/api/projects/ai07/engineering');assert response.status_code==200
    assert all(c['status']!='已通过' for c in response.json()['criteria'])
    assert client.get('/api/documents',params={'path':'SRS.md'}).status_code==200
    assert client.get('/api/documents',params={'path':'../../.env'}).status_code==404
    assert client.get('/api/projects/ai07/events').status_code==200
    assert 'ai07_http_requests_total' in client.get('/metrics').text

def test_worker_plan_is_argv_only_and_paths_are_bounded(tmp_path,monkeypatch):
    from platform_api import worker
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    job={'id':'a'*24,'kind':'generate','payload':payload()}
    plan=worker.command_plan(job);assert plan[0][0]=='preview' and isinstance(plan[0][1],list)
    job['payload']['route']='physics';job['payload']['enhance']=True
    assert [x[0] for x in worker.command_plan(job)]==['physics','blender','enhance']
    job['id']='../../escape'
    with pytest.raises(ValueError):worker.command_plan(job)


def test_label_storage_is_batch_scoped_and_cached_sync_repairs_permissions(client,monkeypatch):
    from platform_api.db import Link
    requests=[]; storages=[]
    def request(config,method,path,**kwargs):
        requests.append((method,path,kwargs))
        if method=='GET': return storages
        if path=='storages/localfiles/':
            storages.append({**kwargs['json'],'id':7}); return storages[-1]
        raise AssertionError('Cached sync must not create another project or duplicate tasks')
    monkeypatch.setattr(integrations,'label_request',request)
    monkeypatch.setenv('AI07_LABEL_STUDIO_PUBLIC_URL','http://localhost:8082')
    with client.factory.begin() as db:
        run=db.scalars(select(Run)).first()
        db.add(Link(run_id=run.id,value={'project_id':11,'task_count':2,'url':'http://internal/projects/11/data'}))
        db.flush()
        value=integrations.label_sync(db,run)
        assert value['url']=='http://localhost:8082/projects/11/data'
        assert storages[0]['path']=='/data/artifacts/demo/room_demo/images'
        assert storages[0]['project']==11 and not storages[0]['recursive_scan']
        integrations.label_sync(db,run)
        assert sum(method=='POST' for method,_,_ in requests)==1


def simulation_payload():
    return {'config':SceneConfig(width=64,height=64).to_dict(),'name':'Simulation contract','idempotency_key':'simulation-contract'}


def test_simulation_input_and_resource_boundary(client):
    p=simulation_payload()
    assert client.post('/api/projects/ai07/simulations/preview-input',json={}).status_code==400
    assert client.get('/api/simulation-engines').json()['engine']=='fds'
    preview=client.post('/api/projects/ai07/simulations/preview-input',json=p)
    assert preview.status_code==200 and "QUANTITY='TEMPERATURE'" in preview.json()['content']
    assert client.get('/api/projects/ai07/simulations').json()==[]
    assert client.post('/api/projects/ai07/simulations',json=p).status_code==401
    p['config']['cell_size_m']=.05
    assert client.post('/api/projects/ai07/simulations/preview-input',json=p).status_code==400
    p=simulation_payload();p['config']['scene_id']='../../escape'
    assert client.post('/api/projects/ai07/simulations/preview-input',json=p).status_code==400


def test_simulation_queue_reuses_permissions_idempotency_and_retry(client):
    login(client);p=simulation_payload()
    first=client.post('/api/projects/ai07/simulations',json=p).json()
    assert first['kind']=='simulation' and not first['payload']['enhance']
    assert client.post('/api/projects/ai07/simulations',json=p).json()['id']==first['id']
    assert client.get(f'/api/simulations/{first["id"]}').status_code==409
    client.post(f'/api/jobs/{first["id"]}/cancel')
    retry=client.post(f'/api/jobs/{first["id"]}/retry').json()
    assert retry['id']!=first['id'] and retry['kind']=='simulation'
    other=client.post('/api/projects',json={'name':'Other simulation project'}).json()
    # A separate project cannot be accessed by an unauthenticated viewer.
    client.post('/api/logout')
    assert client.get(f'/api/projects/{other["id"]}/simulations').status_code==401


def test_simulation_requires_actual_fields_and_frame_hashes(client):
    from fire_modeling.io import write_json,sha256
    from fire_modeling.simulation import SCHEMA,FIELDS
    login(client);j=client.post('/api/projects/ai07/simulations',json=simulation_payload()).json()
    headers={'x-worker-key':(client.test_root/'.cache/platform/worker.key').read_text()}
    claim=client.post('/api/internal/claim',headers=headers).json()
    endpoint=f'/api/internal/jobs/{j["id"]}'; done={'lease':claim['lease'],'status':'succeeded'}
    assert client.post(endpoint,headers=headers,json=done).status_code==400
    root=client.test_root/f'artifacts/platform/jobs/{j["id"]}/physics'
    write_json(root/'execution.json',{'status':'success','last_time_s':5})
    write_json(root/'web/frame.json',{'values':[[20,80],[30,50]],'time_s':1})
    report={'schema':SCHEMA,'status':'success','fields':[{'id':key,'frames':[{'file':'frame.json','time_s':1,'sha256':sha256(root/'web/frame.json')}]} for key in FIELDS]}
    write_json(root/'web/simulation.json',report)
    assert client.post(endpoint,headers=headers,json=done).status_code==200
    assert client.get(f'/api/simulations/{j["id"]}').json()['schema']==SCHEMA
    url=f'/api/simulations/{j["id"]}/fields/temperature/frames/0'
    assert client.get(url).json()['values'][0][1]==80
    assert client.get(url.replace('frames/0','frames/99')).status_code==404
    assert client.get(url.replace('temperature','unknown')).status_code==404
    write_json(root/'web/frame.json',{'values':[[999]],'time_s':1})
    assert client.get(url).status_code==409


def test_simulation_worker_uses_only_fds_stage(tmp_path,monkeypatch):
    from platform_api import worker
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    plan=worker.command_plan({'id':'b'*24,'kind':'simulation','payload':simulation_payload()})
    assert len(plan)==1 and 'simulate-web' in plan[0][1]
    assert 'physics' in plan[0][1] and 'blender' not in plan[0][1] and 'gpu' not in plan[0][1]


def test_firephantom_jump_reports_offline_and_never_proxies_arbitrary_url(client,monkeypatch):
    monkeypatch.setattr(web,'firephantom_status',lambda:{'available':False})
    assert client.get('/api/integrations/firephantom').json()=={'available':False}
    assert client.get('/api/integrations/firephantom/open',follow_redirects=False).status_code==503
    monkeypatch.setattr(web,'firephantom_status',lambda:{'available':True})
    response=client.get('/api/integrations/firephantom/open?url=http://example.org',follow_redirects=False)
    assert response.status_code==307 and response.headers['location']=='http://localhost:8765/'


def test_firephantom_probe_has_timeout_and_no_secret_forwarding(monkeypatch):
    from platform_api import firephantom
    calls=[]
    class Client:
        def __init__(self,**kwargs):calls.append(kwargs)
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def get(self,url):
            calls.append(url)
            return type('Response',(),{'status_code':200})()
    monkeypatch.setattr(firephantom.httpx,'Client',Client)
    monkeypatch.setenv('AI07_FIREPHANTOM_URL','http://firephantom:8765')
    assert firephantom.service_status()['available']
    assert calls==[{'timeout':3,'trust_env':False,'follow_redirects':False},'http://firephantom:8765/health']


def test_firephantom_deployment_session_and_origin_gate(monkeypatch):
    import importlib.util,sys,types
    from fastapi import FastAPI
    stub=types.ModuleType('firephantom.api');backend=FastAPI()
    @backend.get('/health')
    def health():return {'status':'ok'}
    @backend.get('/api/v1/jobs')
    def jobs():return []
    @backend.post('/api/v1/auth/bootstrap')
    def setup():return {'stub':True}
    stub.create_app=lambda:backend
    stub._current_user=lambda request:{'id':1} if request.cookies.get('fp_session')=='test-session' else None
    monkeypatch.setitem(sys.modules,'firephantom.api',stub)
    path=Path(__file__).resolve().parents[1]/'docker/firephantom_entry.py'
    spec=importlib.util.spec_from_file_location('tested_firephantom_entry',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with TestClient(module.app) as c:
        assert c.get('/health').status_code==200
        assert c.get('/api/v1/jobs').status_code==401
        assert c.post('/api/v1/auth/bootstrap',headers={'origin':'https://foreign.example'}).status_code==403
        assert c.post('/api/v1/auth/bootstrap',headers={'origin':'http://localhost:8765'}).status_code==200
        c.cookies.set('fp_session','test-session')
        assert c.get('/api/v1/jobs').status_code==200


def test_editor_fds_duration_and_trajectory_project_boundary(client):
    from fire_modeling.editor_contract import upstream
    login(client)
    raw={'scenario':'房间火灾','origin_fuel':'木材','placements':upstream().catalog_extras()['defaults']['房间火灾'],'viewpoints':['侧视']}
    saved=client.put('/api/projects/ai07/scenes/room',json={'expected_revision':0,'editor_scene':raw}).json()
    body={'scene_id':'room','scene_revision':saved['config']['editor_scene']['revision'],'generation':{'engine':'fds','frame_count':12,'duration_s':20},'idempotency_key':'trajectory-duration'}
    result=client.post('/api/projects/ai07/scene-editor/generate',json=body)
    assert result.status_code==200,result.text
    job=result.json();assert job['payload']['config']['duration_s']==20
    assert job['payload']['config']['output_frames']==12
    route=f"/api/projects/ai07/scene-editor/compat/jobs/{job['id']}/trajectory"
    assert client.get(route).json()['available'] is False
    assert client.get(route+'?download=true').status_code==409
    other=client.post('/api/projects',json={'name':'Other trajectory project'}).json()
    assert client.get(route.replace('/ai07/',f"/{other['id']}/")).status_code==403
    body['generation']['duration_s']=61
    assert client.post('/api/projects/ai07/scene-editor/generate',json=body).status_code==400


def test_outdoor_editor_job_uses_explicit_grid_and_rejects_over_budget(client):
    from fire_modeling.editor_contract import upstream
    login(client)
    raw={'scenario':'林区火灾','origin_fuel':'植被','placements':upstream().catalog_extras()['defaults']['林区火灾'],'viewpoints':['侧视']}
    saved=client.put('/api/projects/ai07/scenes/outdoor',json={'expected_revision':0,'editor_scene':raw})
    assert saved.status_code==200,saved.text
    row=saved.json();assert row['config']['cell_size_m']==.2
    body={'scene_id':'outdoor','scene_revision':1,'generation':{'engine':'fds','frame_count':4,'duration_s':5,'cell_size_m':.2},'idempotency_key':'outdoor-grid'}
    before=len(client.get('/api/projects/ai07/jobs').json())
    rejected=client.post('/api/projects/ai07/scene-editor/generate',json=body)
    assert rejected.status_code==400 and '4,000,000' in rejected.text
    assert len(client.get('/api/projects/ai07/jobs').json())==before
    body['generation']['cell_size_m']=.8
    success=client.post('/api/projects/ai07/scene-editor/generate',json=body)
    assert success.status_code==200,success.text
    config=success.json()['payload']['config']
    assert config['cell_size_m']==.8
    assert __import__('math').prod(round(d/.8) for d in config['room_size_m'])==62500
    # Grid choice belongs to this job; it must not silently rewrite the saved scene.
    scenes=client.get('/api/projects/ai07/scenes').json()
    assert next(s for s in scenes if s['id']=='outdoor')['config']['cell_size_m']==.2


def test_synthesis_plan_auth_coverage_and_cross_project_reuse(client):
    from fire_modeling.editor_contract import upstream
    from fire_modeling.io import write_json,sha256
    login(client)
    raw={'scenario':'房间火灾','origin_fuel':'木材','placements':upstream().catalog_extras()['defaults']['房间火灾'],'viewpoints':['侧视']}
    saved=client.put('/api/projects/ai07/scenes/room',json={'expected_revision':0,'editor_scene':raw}).json()
    body={'scene_id':'room','scene_revision':1,'generation':{'engine':'fds','frame_count':4},'idempotency_key':'synthesis-source'}
    j=client.post('/api/projects/ai07/scene-editor/generate',json=body).json()
    base=client.test_root/f"artifacts/platform/jobs/{j['id']}/physics";base.mkdir(parents=True)
    write_json(base/'volume/volume.json',{'frames':[{'time_s':i,'file':f'{i}.npz','sha256':str(i)*64,'max_hrrpuv_kw_m3':20} for i in (1,2,3)]})
    write_json(base/'handoff.json',{'config':j['payload']['config'],'volume_sha256':sha256(base/'volume/volume.json')})
    with client.factory() as db:
        job=db.get(Job,j['id']);job.status='succeeded';db.commit()
    spec={'source_ids':[j['id']],'options':{'weathers':['clear','fog'],'time_count':2,'modes':['background']}}
    preview=client.post('/api/projects/ai07/synthesis/preview',json=spec)
    assert preview.status_code==200,preview.text
    assert preview.json()['counts']['raw']==4 and preview.json()['counts']['requested_augmented']==4
    submitted=client.post('/api/projects/ai07/synthesis',json={**spec,'plan_hash':preview.json()['plan_hash'],'idempotency_key':'synthesis-create'})
    assert submitted.status_code==200,submitted.text
    assert submitted.json()['kind']=='synthesis'
    group=client.post(f"/api/projects/ai07/scene-editor/compat/jobs/{submitted.json()['id']}/continue",json={'combo_index':0})
    assert group.status_code==200,group.text
    assert group.json()['payload']['plan']['counts']['raw']==2
    assert group.json()['payload']['plan']['counts']['requested_augmented']==2
    assert 'resume_from' not in group.json()['payload']
    assert client.post(f"/api/projects/ai07/scene-editor/compat/jobs/{submitted.json()['id']}/continue",json={'combo_index':999}).status_code==400
    assert client.post('/api/projects/ai07/synthesis',json={**spec,'plan_hash':'changed','idempotency_key':'synthesis-stale'}).status_code==400
    other=client.post('/api/projects',json={'name':'Other synth project'}).json()
    assert client.post(f"/api/projects/{other['id']}/synthesis/preview",json=spec).status_code==400
    assert client.get(f"/api/projects/{other['id']}/synthesis/jobs/{submitted.json()['id']}/result").status_code==404
    assert client.get(f"/api/projects/ai07/synthesis/jobs/{submitted.json()['id']}/files/../../config.json").status_code==404


def test_auto_mesh_template_snapshot_and_manual_override(client):
    path='/api/projects/ai07/scene-editor'
    assert client.post(path+'/templates/corridor').status_code==401
    login(client)
    response=client.post(path+'/templates/corridor')
    assert response.status_code==200,response.text
    saved=response.json();sid=saved['id']
    rec=client.post(path+'/grid-recommendation',json={'scene_id':sid,'editor_scene':saved['config']['editor_scene'],'temperature':25})
    assert rec.status_code==200,rec.text
    dx=rec.json()['recommended']['cell_size_m']
    body={'scene_id':sid,'scene_revision':1,'generation':{'engine':'fds','mesh_policy':'auto','cell_size_m':2,'duration_s':4,'frame_count':4,'temperature':25},'idempotency_key':'auto-grid-test'}
    result=client.post(path+'/generate',json=body)
    assert result.status_code==200,result.text
    payload=result.json()['payload']
    assert payload['config']['cell_size_m']==dx and payload['mesh_recommendation']['schema']=='ai07.mesh-recommendation.v1'
    assert client.get('/api/projects/ai07/scenes').json()[-1]['config']==saved['config']
    body['generation'].update(mesh_policy='manual',cell_size_m=.2);body['idempotency_key']='manual-grid-test'
    manual=client.post(path+'/generate',json=body)
    assert manual.status_code==200,manual.text
    assert manual.json()['payload']['config']['cell_size_m']==.2
    assert client.post(path+'/templates/missing').status_code==400
    assert client.post(path+'/grid-recommendation',json={'scene_id':sid,'temperature':999}).status_code==400


def test_fds_weather_batch_freezes_independent_physical_conditions(client,monkeypatch):
    login(client);path='/api/projects/ai07/scene-editor'
    scene=client.post(path+'/templates/wood-yard').json()
    body={'scene_id':scene['id'],'scene_revision':1,'generation':{'engine':'fds','mesh_policy':'auto','weathers':['晴','雨'],'humidity':80,'temperature':25,'wind_speed':1,'rain_mm_h':10,'rain_diameter_um':800,'frame_count':4,'duration_s':4},'idempotency_key':'weather-batch-contract'}
    response=client.post(path+'/generate',json=body)
    assert response.status_code==200,response.text
    ids=response.json()['batch_jobs'];assert len(ids)==2
    jobs={r['id']:r for r in client.get('/api/projects/ai07/jobs').json()}
    dry,wet=[jobs[i]['payload'] for i in ids]
    assert dry['config']['rain_mm_h']==0 and wet['config']['rain_mm_h']==10
    assert wet['config']['ambient_relative_humidity_pct']==80 and wet['config']['wind_m_s']==1
    assert wet['config']['physical_weather']=='雨' and wet['config']['rain_diameter_um']==800
    assert wet['scene_revision']==dry['scene_revision']==1
    assert client.post(path+'/generate',json=body).json()['batch_jobs']==ids
    body['generation']['weathers']=['雨','雪']
    assert client.post(path+'/generate',json=body).status_code==400
    assert len(client.get('/api/projects/ai07/jobs').json())==2
    import fire_modeling.trajectory as temporal
    monkeypatch.setattr(temporal,'trajectory',lambda root:{'available':True,'frames':[{'time_s':0,'fds_next':{'target_time_s':1}}]})
    trajectory_path=path+'/compat/jobs/'+ids[0]+'/trajectory'
    assert 'fds_next' in client.get(trajectory_path).json()['frames'][0]
    assert 'fds_next' not in client.get(trajectory_path+'?download=true').json()['frames'][0]


def test_register_self_service_contract(client):
    response=client.post('/api/register',json={'username':'self_tester','password':'password-123456','role':'tester'})
    assert response.status_code==200 and response.json()['role']=='tester'
    assert client.get('/api/auth').json()['user']['username']=='self_tester'
    assert client.get('/api/projects').json()[0]['id']=='ai07'
    assert client.post('/api/register',json={'username':'self_tester','password':'password-123456'}).status_code==409
    assert client.post('/api/register',json={'username':'self_tester2','password':'password-123456','role':'root'}).status_code==400
    assert client.post('/api/register',json={'username':'self_tester3','password':'short'}).status_code==400
    assert client.post('/api/register',json={'username':'bad name!','password':'password-123456'}).status_code==400
    assert client.post('/api/register',json={'username':'self_default','password':'password-123456'}).json()['role']=='tester'


def test_review_timing_contract(client):
    login(client)
    samples=client.get('/api/projects/ai07/samples').json()
    target=next(s for s in samples if s['review_status']=='pending')
    body={'sample_id':target['sample_id'],'boxes':target['boxes'],'decision':'approved',
          'timing':{'active_s':12.4,'mode':'candidate'}}
    assert client.post('/api/runs/'+target['run_id']+'/review',json=body).status_code==200
    stats=client.get('/api/projects/ai07/timing').json()
    assert stats['candidate']['count']>=1
    assert stats['candidate']['mean_active_s_per_sample']>0
    bad=dict(body,timing={'active_s':-3,'mode':'candidate'})
    assert client.post('/api/runs/'+target['run_id']+'/review',json=bad).status_code in (200,400)  # 非法计时不写入但不崩溃
    stats2=client.get('/api/projects/ai07/timing').json()
    assert stats2['candidate']['count']==stats['candidate']['count']
