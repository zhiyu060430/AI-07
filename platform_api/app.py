"""Local research workbench API. Heavy algorithms run in an outbound host worker."""
from contextlib import asynccontextmanager
import hashlib
import hmac
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import threading
import time
import zipfile

from fastapi import FastAPI, Request, Response, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from sqlalchemy import select, func, update
from sqlalchemy.exc import IntegrityError

from .db import ROOT, STATE, engine, Session, User, Login, Project, Member, Scene, Job, Run, Version, Experiment, Event, Setting
from .services import initialize, uid, audit, root_for, sample_views, run_view, job_view, hrr_curve, engineering, DEFAULT_PROJECT
from .integrations import dvc_publish, review_lock, label_config, label_sync, label_pull
from fire_modeling.config import SceneConfig
from fire_modeling.dataset import review, import_coco
from fire_modeling.io import read_json, write_json, resolve_artifact
from fire_modeling.quality import inspect_run
from fire_modeling.evaluation import physical_error, detection_metrics
from fire_modeling.physics import fds_input
from fire_modeling.simulation import load_simulation, FIELDS
from fire_modeling.io import sha256
from .firephantom import service_status as firephantom_status, PUBLIC_URL as FIREPHANTOM_PUBLIC_URL

REQUESTS={'total':0,'errors':0}; started=time.time(); setup_lock=threading.Lock()
ADMIN_ROLES={'architect'}
WRITERS={'product','architect','developer','tester','ops'}
BUILDERS={'product','architect','developer'}

@asynccontextmanager
async def lifespan(app):
    initialize(); yield

app=FastAPI(title='AI-07 火焰数据工作台',version='0.2.0',lifespan=lifespan)

@app.middleware('http')
async def protect_origin(request,call_next):
    REQUESTS['total']+=1
    path=request.url.path
    public_auth={'/api/auth','/api/login','/api/logout','/api/setup','/api/register','/api/health'}
    protected=path.startswith('/api/') or path in {'/docs','/openapi.json','/metrics'}
    # Internal worker endpoints enforce their own shared-key authentication.
    if os.getenv('AI07_REQUIRE_LOGIN')=='1' and protected and path not in public_auth and not path.startswith('/api/internal/'):
        with Session() as db:
            if current_user(request,db) is None:
                return JSONResponse({'detail':'请登录工作区后访问平台数据'},status_code=401,headers={'Cache-Control':'no-store'})
    if request.method not in {'GET','HEAD','OPTIONS'}:
        origin=request.headers.get('origin')
        allowed={f'http://{h}:{p}' for h in ('localhost','127.0.0.1') for p in (3000,5173,8090,8088)}
        allowed.update(filter(None,os.getenv('AI07_ALLOWED_ORIGINS','').split(',')))
        if origin and origin not in allowed:
            return JSONResponse({'detail':'请求来源不受信任'},status_code=403)
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='same-origin'
    response.headers['Cache-Control']='no-store'
    if response.status_code>=400: REQUESTS['errors']+=1
    return response

@app.exception_handler(ValueError)
async def value_error(request,exc): return JSONResponse({'detail':str(exc)},status_code=400)

@app.exception_handler(RuntimeError)
async def runtime_error(request,exc): return JSONResponse({'detail':str(exc)},status_code=409)

def db_session():
    with Session() as db: yield db

def current_user(request:Request,db=Depends(db_session)):
    token=request.cookies.get('ai07_session','')
    record=db.get(Login,hashlib.sha256(token.encode()).hexdigest()) if token else None
    return db.get(User,record.user_id) if record and record.expires>time.time() else None

def user_view(user): return {'id':user.id,'username':user.username,'role':user.role} if user else None

def access(db,user,project,write=False,roles=None):
    if not db.get(Project,project): raise HTTPException(404,'项目不存在')
    if user is None:
        if project==DEFAULT_PROJECT and not write: return
        raise HTTPException(401,'请登录工作区后继续')
    if not db.get(Member,(project,user.id)): raise HTTPException(403,'没有该项目的访问权限')
    if write and user.role not in ((roles or WRITERS)|ADMIN_ROLES): raise HTTPException(403,'当前角色没有该操作权限')

def scoped(db,model,id,user,write=False,roles=None):
    row=db.get(model,id)
    if not row: raise HTTPException(404,'记录不存在')
    access(db,user,row.project_id,write,roles); return row

def password_hash(password,salt=None):
    if not isinstance(password,str) or len(password)<10 or len(password)>256: raise ValueError('密码需为10至256位')
    salt=salt or secrets.token_hex(16)
    return salt+':'+hashlib.pbkdf2_hmac('sha256',password.encode(),salt.encode(),240000).hex()

def login_cookie(response,user,db):
    token=secrets.token_urlsafe(40)
    db.add(Login(token_hash=hashlib.sha256(token.encode()).hexdigest(),user_id=user.id,expires=time.time()+43200))
    response.set_cookie('ai07_session',token,httponly=True,samesite='strict',max_age=43200,secure=os.getenv('AI07_SECURE_COOKIE')=='1')

@app.get('/api/auth')
def auth(user=Depends(current_user),db=Depends(db_session)):
    first=db.scalar(select(func.count()).select_from(User))==0
    return {'user':user_view(user),'setup_required':first,
            'require_login':os.getenv('AI07_REQUIRE_LOGIN')=='1',
            'registration_open':first or os.getenv('AI07_MANAGED_ACCOUNTS')!='1'}

@app.post('/api/setup')
def setup(data:dict,response:Response,db=Depends(db_session)):
    with setup_lock:
        if db.scalar(select(func.count()).select_from(User)): raise HTTPException(409,'管理员已创建，请登录')
        username=str(data.get('username','')).strip()
        if not re.fullmatch(r'[A-Za-z0-9_-]{3,32}',username): raise ValueError('用户名需为3至32位字母、数字或下划线')
        user=User(id=uid(),username=username,password_hash=password_hash(data.get('password')),role='architect')
        db.add(user); db.add(Member(project_id=DEFAULT_PROJECT,user_id=user.id))
        db.get(Project,DEFAULT_PROJECT).owner_id=user.id
        login_cookie(response,user,db); audit(db,DEFAULT_PROJECT,user.username,'创建本地管理员'); db.commit()
        return user_view(user)

@app.post('/api/login')
def login(data:dict,response:Response,db=Depends(db_session)):
    user=db.scalar(select(User).where(User.username==data.get('username')))
    try: valid=user and hmac.compare_digest(password_hash(data.get('password'),user.password_hash.split(':')[0]),user.password_hash)
    except ValueError: valid=False
    if not valid: raise HTTPException(401,'用户名或密码不正确')
    login_cookie(response,user,db); db.commit(); return user_view(user)

@app.post('/api/register')
def register(data:dict,response:Response,db=Depends(db_session)):
    if os.getenv('AI07_MANAGED_ACCOUNTS')=='1':
        if db.scalar(select(func.count()).select_from(User)):
            raise HTTPException(403,'请由管理员在工程与验收中创建账户并分配角色')
        return setup(data,response,db)
    username=str(data.get('username','')).strip()
    role=data.get('role') or 'tester'
    if not re.fullmatch(r'[A-Za-z0-9_-]{3,32}',username): raise ValueError('用户名需为3至32位字母、数字或下划线')
    if role not in WRITERS: raise ValueError('所选角色不存在')
    if db.scalar(select(User).where(User.username==username)): raise HTTPException(409,'用户名已被注册')
    user=User(id=uid(),username=username,password_hash=password_hash(data.get('password')),role=role)
    db.add(user); db.add(Member(project_id=DEFAULT_PROJECT,user_id=user.id))
    login_cookie(response,user,db); audit(db,DEFAULT_PROJECT,username,'自助注册'); db.commit()
    return user_view(user)

@app.post('/api/logout')
def logout(request:Request,response:Response,db=Depends(db_session)):
    token=request.cookies.get('ai07_session',''); record=db.get(Login,hashlib.sha256(token.encode()).hexdigest())
    if record: db.delete(record); db.commit()
    response.delete_cookie('ai07_session'); return {'ok':True}

@app.get('/api/projects')
def projects(user=Depends(current_user),db=Depends(db_session)):
    ids=list(db.scalars(select(Member.project_id).where(Member.user_id==user.id))) if user else [DEFAULT_PROJECT]
    return [{'id':p.id,'name':p.name} for p in db.scalars(select(Project).where(Project.id.in_(ids)))]

@app.post('/api/projects')
def create_project(data:dict,user=Depends(current_user),db=Depends(db_session)):
    if not user or user.role not in ADMIN_ROLES: raise HTTPException(403,'仅架构师（管理员）可创建项目')
    name=str(data.get('name','')).strip()
    if not 1<=len(name)<=80: raise ValueError('请输入1至80字的项目名称')
    id=uid(); db.add(Project(id=id,name=name,owner_id=user.id)); db.add(Member(project_id=id,user_id=user.id)); db.commit()
    return {'id':id,'name':name}

@app.get('/api/projects/{project}/users')
def users(project:str,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project,True,ADMIN_ROLES)
    ids=list(db.scalars(select(Member.user_id).where(Member.project_id==project)))
    return [user_view(u) for u in db.scalars(select(User).where(User.id.in_(ids)))]

@app.post('/api/projects/{project}/users')
def create_user(project:str,data:dict,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project,True,ADMIN_ROLES)
    name=str(data.get('username','')); role=data.get('role')
    if not re.fullmatch(r'[A-Za-z0-9_-]{3,32}',name) or role not in WRITERS: raise ValueError('用户名或角色不合法')
    if db.scalar(select(User).where(User.username==name)): raise ValueError('用户名已存在')
    row=User(id=uid(),username=name,password_hash=password_hash(data.get('password')),role=role)
    db.add(row); db.add(Member(project_id=project,user_id=row.id)); audit(db,project,user.username,'添加项目成员',username=name,role=role); db.commit()
    return user_view(row)

@app.get('/api/projects/{project}/scenes')
def scenes(project:str,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project)
    return [{'id':s.id,'name':s.name,'config':s.config} for s in db.scalars(select(Scene).where(Scene.project_id==project))]

@app.post('/api/projects/{project}/scenes')
def save_scene(project:str,data:dict,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project,True,BUILDERS)
    from .scene_editor import assert_assets
    assert_assets(ROOT,project,data.get('config',{}).get('editor_scene'));config=SceneConfig.from_dict(data['config'])
    row=Scene(id=uid(),name=str(data.get('name',config.scene_id))[:80],project_id=project,config=config.to_dict())
    db.add(row); audit(db,project,user.username,'保存场景',scene=row.id); db.commit(); return {'id':row.id}

@app.get('/api/projects/{project}/scene-editor/catalog')
def editor_catalog(project:str,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project)
    from .scene_editor import catalog,uploads
    return {**catalog(ROOT,project),'uploads':uploads(ROOT,project)}

@app.post('/api/projects/{project}/scene-editor/preview')
def editor_preview(project:str,data:dict,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project)
    from .scene_editor import assert_assets
    from fire_modeling.editor_contract import normalize_scene,upstream
    assert_assets(ROOT,project,data);s=normalize_scene(data)
    return upstream().layout_preview(s['scenario'],s['placements'],s['origin_fuel'],origin=s['origin'],cameras=s['cameras'],
        gas_domain_override=s['gas_domain'],viewpoints=s['viewpoints'],lights=s['lights'],simulation=s['simulation'],
        structures=s['structures'],domain=s['domain'],frame_count=1)

@app.put('/api/projects/{project}/scenes/{id}')
def update_editor_scene(project:str,id:str,data:dict,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project,True,BUILDERS)
    row=db.scalar(select(Scene).where(Scene.id==id).with_for_update())
    if not row or row.project_id!=project:raise HTTPException(404,'场景不存在')
    from .scene_editor import assert_assets
    from fire_modeling.editor_contract import bind_config
    if not isinstance(data.get('editor_scene'),dict):raise ValueError('请提供完整的三维场景对象')
    assert_assets(ROOT,project,data.get('editor_scene'))
    current=(row.config.get('editor_scene') or {}).get('revision',0)
    if data.get('expected_revision')!=current:raise HTTPException(409,'场景已由其他人修改，请重新载入后再保存')
    raw={**data['editor_scene'],'revision':current+1}
    config=bind_config(row.config,raw)
    row.config=config.to_dict();row.name=str(data.get('name',row.name))[:80]
    audit(db,project,user.username,'更新三维场景',scene=id,revision=current+1)
    db.commit();return {'id':id,'name':row.name,'config':row.config}

@app.post('/api/projects/{project}/scene-assets')
async def upload_scene_asset(project:str,file:UploadFile=File(...),user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project,True,BUILDERS)
    if not str(file.filename).lower().endswith('.glb'):raise ValueError('请上传GLB模型')
    from .scene_editor import store_asset
    with review_lock:record=store_asset(ROOT,project,await file.read(8*1024*1024+1),file.filename)
    audit(db,project,user.username,'上传场景模型',asset=record['asset_id']);db.commit();return record

@app.get('/api/projects/{project}/scene-assets/{kind}/{filename}')
def get_scene_asset(project:str,kind:str,filename:str,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project)
    from .scene_editor import asset_file
    file=asset_file(ROOT,project,kind,filename)
    if not file.is_file():raise HTTPException(404,'模型文件不存在')
    return FileResponse(file,media_type='model/gltf-binary')

@app.get('/api/projects/{project}/overview')
def overview(project:str,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project); runs=list(db.scalars(select(Run).where(Run.project_id==project).order_by(Run.created.desc())))
    all_samples=[s for r in runs for s in sample_views(r)]
    jobs=list(db.scalars(select(Job).where(Job.project_id==project).order_by(Job.created.desc()).limit(12)))
    return {'runs':[run_view(r) for r in runs],'samples':len(all_samples),'approved':sum(s['review_status']=='approved' for s in all_samples),
            'pending':sum(s['review_status']=='pending' for s in all_samples),'enhanced':sum(s['source']=='diffusion' for s in all_samples),
            'real':sum(s['source']=='real' for s in all_samples),'running':sum(j.status in {'queued','running'} for j in jobs),
            'jobs':[job_view(j) for j in jobs],'featured':sorted(all_samples,key=lambda s:s['route']!='physics')[:6],
            'curve':next((c for r in runs if (c:=hrr_curve(r))),[])}

@app.get('/api/projects/{project}/samples')
def samples(project:str,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project)
    return [s for r in db.scalars(select(Run).where(Run.project_id==project)) for s in sample_views(r)]

@app.get('/api/runs/{id}/samples/{sample_id}/{kind}')
def image(id:str,sample_id:str,kind:str,user=Depends(current_user),db=Depends(db_session)):
    run=scoped(db,Run,id,user); sample=next((s for s in sample_views(run) if s['sample_id']==sample_id),None)
    if not sample or kind not in {'image','mask'} or not sample.get(kind): raise HTTPException(404,'图像不存在')
    return FileResponse(resolve_artifact(root_for(run),sample[kind]),media_type='image/png')

@app.post('/api/runs/{id}/review')
def review_sample(id:str,data:dict,user=Depends(current_user),db=Depends(db_session)):
    run=scoped(db,Run,id,user,True,{'product','tester'})
    with review_lock: review(root_for(run),data['sample_id'],user.username,data['decision'],data.get('boxes'),data.get('timing'))
    audit(db,run.project_id,user.username,'审核样本',sample=data['sample_id'],decision=data['decision']); db.commit()
    return {'ok':True}

@app.get('/api/projects/{project}/timing')
def timing_stats(project:str,user=Depends(current_user),db=Depends(db_session)):
    scoped_projects=db.scalars(select(Run).where(Run.project_id==project))
    modes:dict[str,dict]={}
    for run in scoped_projects:
        path=root_for(run)/'reviews.json'
        if not path.exists(): continue
        for revision in read_json(path).get('history',[]):
            timing=revision.get('timing')
            if not timing: continue
            mode=timing['mode']; slot=modes.setdefault(mode,{'count':0,'total_active_s':0.0,'runs':set()})
            slot['count']+=1; slot['total_active_s']+=timing['active_s']; slot['runs'].add(run.id)
    payload={mode:{'count':v['count'],'total_active_s':round(v['total_active_s'],1),
                   'mean_active_s_per_sample':round(v['total_active_s']/v['count'],1) if v['count'] else 0,
                   'runs':sorted(v['runs'])} for mode,v in modes.items()}
    if not any(v['count'] for v in modes.values()): payload['note']='尚无人工标注计时记录；在样本页框选审核会自动累计有效操作时长。'
    return payload

@app.post('/api/runs/{id}/quality')
def inspect(id:str,user=Depends(current_user),db=Depends(db_session)):
    run=scoped(db,Run,id,user,True); root=root_for(run)
    report=inspect_run(root,read_json(root/'manifest.json')); write_json(root/'quality.json',report)
    audit(db,run.project_id,user.username,'运行结构质检',count=report['count']); db.commit(); return report

@app.get('/api/runs/{id}/label-studio')
def label_studio_status(id:str,user=Depends(current_user),db=Depends(db_session)):
    from .label_bridge import status
    return status(db,scoped(db,Run,id,user))

@app.post('/api/runs/{id}/label-studio/{action}')
def label_studio(id:str,action:str,user=Depends(current_user),db=Depends(db_session)):
    run=scoped(db,Run,id,user,True,{'product','tester'})
    if action=='sync': result=label_sync(db,run)
    elif action=='push':
        from .label_bridge import push
        result=push(db,run)
    elif action=='pull':
        from .label_bridge import acknowledge_pull
        result={'imported':label_pull(db,run,user.username)};acknowledge_pull(db,run)
    else: raise HTTPException(404,'未知操作')
    audit(db,run.project_id,user.username,'Label Studio '+action); db.commit(); return result

@app.get('/api/projects/{project}/jobs')
def jobs(project:str,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project)
    return [job_view(j) for j in db.scalars(select(Job).where(Job.project_id==project).order_by(Job.created.desc()).limit(100))]

def enqueue(db,project,user,payload,kind='generate'):
    if os.getenv('AI07_GPU_ENABLED')=='0' and (
        kind in {'fid','detector-pair'} or (payload.get('enhance') and payload.get('route')=='physics')
    ):
        raise ValueError('当前部署未启用GPU环境；请关闭扩散增强，或使用 start.ps1 -Gpu / start.sh --gpu 准备GPU依赖后重试')
    key=str(payload.get('idempotency_key',''))
    if not 8<=len(key)<=100: raise ValueError('请求需包含8至100位幂等键')
    old=db.scalar(select(Job).where(Job.project_id==project,Job.idempotency_key==key))
    if old: return job_view(old)
    row=Job(id=uid(),project_id=project,idempotency_key=key,payload=payload,kind=kind)
    db.add(row); audit(db,project,user.username,'提交任务',job=row.id,kind=kind)
    try: db.commit()
    except IntegrityError:
        db.rollback(); row=db.scalar(select(Job).where(Job.project_id==project,Job.idempotency_key==key))
    return job_view(row)

@app.post('/api/projects/{project}/jobs')
def create_job(project:str,data:dict,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project,True,BUILDERS)
    if data.get('source_simulation_id'):
        source=scoped(db,Job,str(data['source_simulation_id']),user)
        if source.project_id!=project or source.kind!='video' or source.status!='succeeded':raise ValueError('请选择同项目已完成的视频仿真任务')
        data={**data,'config':source.payload['config'],'scene_id':source.payload['scene_id'],'route':'physics'}
    from .scene_editor import assert_assets
    assert_assets(ROOT,project,data.get('config',{}).get('editor_scene'))
    from fire_modeling.editor_contract import bind_config
    config=bind_config(data['config'],data['config']['editor_scene']) if data['config'].get('editor_scene') else SceneConfig.from_dict(data['config'])
    data={**data,'config':config.to_dict()};route=data.get('route','physics'); count=data.get('count',2)
    if route not in {'physics','preview'} or type(count)!=int or not 1<=count<=100: raise ValueError('路线或数量不合法（每批1至100张）')
    if route=='physics' and (config.width>768 or config.height>768 or config.duration_s>60): raise ValueError('本机交互批次暂限制≤768px、FDS≤60秒；大批次请离线验证资源后运行')
    if route=='physics': simulation_config(data)
    return enqueue(db,project,user,{**data,'config':config.to_dict(),'route':route,'enhance':bool(data.get('enhance',True)),
                                    'pipeline_schema':'fds-volume-v1' if route=='physics' else 'preview-v1'})

@app.post('/api/jobs/{id}/{action}')
def job_action(id:str,action:str,user=Depends(current_user),db=Depends(db_session)):
    job=scoped(db,Job,id,user,True,BUILDERS|{'ops','tester'})
    if action=='cancel':
        if job.status not in {'queued','running'}: raise ValueError('该任务已经结束')
        job.cancel_requested=True
        if job.status=='queued': job.status='cancelled'; job.stage='cancelled'
        audit(db,job.project_id,user.username,'请求取消',job=id); db.commit(); return job_view(job)
    if action=='retry':
        if job.status not in {'failed','interrupted','cancelled'}: raise ValueError('只能重试失败或中断任务')
        return enqueue(db,job.project_id,user,{**job.payload,'idempotency_key':uid(),'retry_of':job.id,**({'resume_from':job.id} if job.kind in {'video','synthesis'} else {})},job.kind)
    raise HTTPException(404,'未知任务操作')

@app.get('/api/simulation-engines')
def simulation_engines():
    return {'engine':'fds','solver_repository':'https://github.com/firemodels/fds',
            'reader_repository':'https://github.com/FireDynamics/fdsreader','reader_version':'1.14.0',
            'transport':'HTTP JSON → persistent job → Compose CLI → FDS files → fdsreader → JSON',
            'max_cells':120000,'max_duration_s':60,'schema':'ai07.simulation.v1'}

@app.get('/api/integrations/firephantom')
def firephantom_integration():
    return firephantom_status()

@app.get('/api/integrations/firephantom/open')
def open_firephantom():
    if not firephantom_status()['available']:
        raise HTTPException(503,'FirePhantom未启动，请在项目根目录运行 scripts/start-firephantom.ps1')
    return RedirectResponse(FIREPHANTOM_PUBLIC_URL,status_code=307)

def simulation_config(data):
    if not isinstance(data.get('config'),dict): raise ValueError('请提供完整的config场景配置对象')
    from fire_modeling.editor_contract import bind_config
    config=bind_config(data['config'],data['config']['editor_scene']) if data['config'].get('editor_scene') else SceneConfig.from_dict(data['config'])
    cells=math.prod(round(n/config.cell_size_m) for n in config.room_size_m)
    if cells>120000:
        raise ValueError(f'未创建作业：当前FDS网格为{cells:,}个，超过本机120,000个上限。请在渲染设置选择推荐网格或缩小计算域；缩短时长不能减少网格数量')
    if config.duration_s>60:raise ValueError('未创建作业：FDS求解时长不能超过60秒')
    return config

@app.post('/api/projects/{project}/simulations/preview-input')
def preview_simulation_input(project:str,data:dict,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project)
    from .scene_editor import assert_assets
    assert_assets(ROOT,project,data.get('config',{}).get('editor_scene'))
    config=simulation_config(data)
    return {'engine':'fds','filename':config.scene_id+'.fds','content':fds_input(config)}

@app.post('/api/projects/{project}/simulation-videos')
def create_simulation_video(project:str,data:dict,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project,True,BUILDERS)
    scene=scoped(db,Scene,str(data.get('scene_id','')),user)
    if scene.project_id!=project or not scene.config.get('editor_scene'):raise ValueError('请先保存三维编辑场景')
    revision=scene.config['editor_scene']['revision']
    if data.get('scene_revision')!=revision:raise HTTPException(409,'场景已改变，请重新载入后提交')
    overrides=data.get('parameters') or {}
    if not isinstance(overrides,dict) or set(overrides)-{'cell_size_m','duration_s','peak_hrr_kw','width','height','render_samples'}:raise ValueError('未知仿真参数')
    config=simulation_config({'config':{**scene.config,**overrides}})
    if config.duration_s>12 or max(config.width,config.height)>768:raise ValueError('交互视频限12秒、768px；可选择较短时长或较小图像')
    from .scene_editor import assert_assets
    assert_assets(ROOT,project,config.editor_scene)
    return enqueue(db,project,user,{'scene_id':scene.id,'scene_revision':revision,'config':config.to_dict(),
        'name':scene.name+' · 仿真视频','route':'physics','enhance':False,
        'idempotency_key':data.get('idempotency_key','')},kind='video')

@app.get('/api/simulation-videos/{id}')
def simulation_video_report(id:str,user=Depends(current_user),db=Depends(db_session)):
    job=scoped(db,Job,id,user)
    if job.kind not in {'video','editor'} or job.status!='succeeded':raise HTTPException(409,'仿真视频尚未完成')
    report=read_json(ROOT/f'artifacts/platform/jobs/{id}/video/video.json')
    from fire_modeling.video import playback_metadata
    return {**report,'id':id,'scene_id':job.payload['scene_id'],
        'videos':[{**playback_metadata(v),'url':f'/api/simulation-videos/{id}/files/{i}'} for i,v in enumerate(report['videos'])]}

@app.get('/api/simulation-videos/{id}/files/{index}')
def simulation_video_file(id:str,index:int,user=Depends(current_user),db=Depends(db_session)):
    job=scoped(db,Job,id,user)
    if job.kind not in {'video','editor'} or job.status!='succeeded':raise HTTPException(409,'仿真视频尚未完成')
    root=ROOT/f'artifacts/platform/jobs/{id}/video';report=read_json(root/'video.json')
    if not 0<=index<len(report['videos']):raise HTTPException(404,'视频不存在')
    item=report['videos'][index];path=resolve_artifact(root,item['file'])
    if sha256(path)!=item['sha256']:raise HTTPException(409,'视频校验失败')
    return FileResponse(path,media_type='video/mp4',filename=f'{id}-{index}.mp4',content_disposition_type='inline')

@app.post('/api/projects/{project}/simulations')
def create_simulation(project:str,data:dict,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project,True,BUILDERS)
    from .scene_editor import assert_assets
    assert_assets(ROOT,project,data.get('config',{}).get('editor_scene'))
    config=simulation_config(data)
    return enqueue(db,project,user,{'name':str(data.get('name','FDS物理仿真'))[:80],
                                   'config':config.to_dict(),'route':'physics','enhance':False,
                                   'idempotency_key':data.get('idempotency_key','')},kind='simulation')

@app.get('/api/projects/{project}/simulations')
def list_simulations(project:str,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project)
    rows=db.scalars(select(Job).where(Job.project_id==project,Job.kind.in_(['simulation','generate'])).order_by(Job.created.desc()).limit(100))
    return [job_view(j) for j in rows if j.payload.get('route')=='physics' and (j.kind=='simulation' or (ROOT/f'artifacts/platform/jobs/{j.id}/physics/web/simulation.json').is_file())]

def simulation_directory(id,user,db):
    job=scoped(db,Job,id,user)
    if job.kind not in {'simulation','generate','video'} or job.payload.get('route')!='physics': raise HTTPException(404,'这不是物理仿真任务')
    if job.status!='succeeded': raise HTTPException(409,'仿真结果尚未完成；请查看任务日志')
    return resolve_artifact(ROOT/'artifacts/platform/jobs',id+'/physics')

@app.get('/api/simulations/{id}')
def simulation_result(id:str,user=Depends(current_user),db=Depends(db_session)):
    directory=simulation_directory(id,user,db)
    report=load_simulation(directory)
    volume_path=directory/'volume/volume.json'
    volume=read_json(volume_path) if volume_path.is_file() else {}
    movement=[{'time_s':f['time_s'],'xyz_m':f['combustion_centroid_m']} for f in volume.get('frames',[]) if f.get('combustion_centroid_m')]
    return {**report,'id':id,'combustion_movement':movement,'input_url':f'/api/simulations/{id}/input',
            'result_url':f'/api/simulations/{id}'}

@app.get('/api/simulations/{id}/input')
def simulation_input_download(id:str,user=Depends(current_user),db=Depends(db_session)):
    directory=simulation_directory(id,user,db); config=db.get(Job,id).payload['config']
    return FileResponse(resolve_artifact(directory,config['scene_id']+'.fds'),media_type='text/plain',filename=config['scene_id']+'.fds')

@app.get('/api/simulations/{id}/fields/{field}/frames/{frame}')
def simulation_frame(id:str,field:str,frame:int,user=Depends(current_user),db=Depends(db_session)):
    directory=simulation_directory(id,user,db)
    if field not in FIELDS: raise HTTPException(404,'未知物理场')
    meta=next(f for f in load_simulation(directory)['fields'] if f['id']==field)
    if not 0<=frame<len(meta['frames']): raise HTTPException(404,'时间帧不存在')
    entry=meta['frames'][frame]; file=resolve_artifact(directory/'web',entry['file'])
    if sha256(file)!=entry['sha256']: raise HTTPException(409,'场数据校验失败，禁止显示被修改的结果')
    return FileResponse(file,media_type='application/json')

@app.get('/api/projects/{project}/versions')
def versions(project:str,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project)
    return [{k:getattr(v,k) for k in ('id','name','run_id','status','result','created')} for v in db.scalars(select(Version).where(Version.project_id==project).order_by(Version.created.desc()))]

@app.post('/api/projects/{project}/versions')
def publish_version(project:str,data:dict,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project,True,{'product','architect','tester'})
    run=scoped(db,Run,data['run_id'],user)
    if run.project_id!=project: raise HTTPException(403,'批次不属于当前项目')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}',str(data.get('name',''))): raise ValueError('版本名只能包含字母、数字、点、横线和下划线')
    if not data.get('research_snapshot'): raise ValueError('当前仅允许明确确认的研究快照，不能冒充课程正式验收版本')
    if db.scalar(select(Version).where(Version.project_id==project,Version.name==data['name'])): raise ValueError('版本名已经存在，旧版本不可覆盖')
    row=Version(id=uid(),project_id=project,name=data['name'],run_id=run.id); vid=row.id
    db.add(row); db.commit()
    def settle(status:str,result):
        # 独立会话＋先取局部 id：长发布期间客户端断开会使请求会话与 ORM 实例失效，避免状态永远停在 staging
        with Session() as fresh:
            target=fresh.get(Version,vid)
            if target: target.status=status; target.result=result; fresh.commit()
    try:
        result=dvc_publish(run,vid,row.name)
        settle('published',result)
        with Session() as fresh:
            audit(fresh,project,user.username,'发布DVC研究快照',version=vid,commit=result['commit']); fresh.commit()
        return {'id':vid,'status':'published','result':result}
    except Exception as exc:
        settle('failed',{'error':str(exc)}); raise RuntimeError(str(exc)) from exc

@app.get('/api/versions/{id}/download')
def download(id:str,user=Depends(current_user),db=Depends(db_session)):
    row=scoped(db,Version,id,user)
    if row.status!='published': raise HTTPException(409,'版本尚未发布成功')
    return FileResponse(ROOT/f'artifacts/platform/archives/{row.id}.zip',filename=f'{row.name}-COCO-YOLO.zip')

@app.post('/api/projects/{project}/import-coco')
async def upload_coco(project:str,file:UploadFile|None=File(None),source_url:str=Form(''),license_note:str=Form(''),user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project,True,BUILDERS|{'tester'})
    import asyncio
    from . import remote_import
    url=source_url.strip(); note=license_note.strip()
    has_file=bool(file and (file.filename or '').strip())
    content=None; run_name=None; sampling=None
    if has_file:
        content=await file.read(50*1024*1024+1)
        if len(content)>50*1024*1024: raise ValueError('单次上传不超过50MB')
        if not url: raise ValueError('上传COCO ZIP时必须填写原始来源链接')
        if not note: raise ValueError('上传COCO ZIP时必须填写授权或许可证说明')
    elif not url:
        raise ValueError('请上传COCO ZIP或填写可下载的真实数据链接（二选一）')
    id=uid(); base=ROOT/'artifacts/platform/imports'/id; unpack=base/'unpack'; unpack.mkdir(parents=True)
    try:
        if content is not None:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                if sum(i.file_size for i in z.infolist())>150*1024*1024 or len(z.infolist())>3000: raise ValueError('解压尺寸或文件数量超出限制')
                for item in z.infolist():
                    dest=(unpack/item.filename.replace('\\','/')).resolve()
                    if not dest.is_relative_to(unpack.resolve()) or item.flag_bits&1: raise ValueError('压缩包包含非法路径或加密文件')
                    if item.is_dir(): continue
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(item) as source,dest.open('wb') as target: target.write(source.read())
            run_name=(file.filename or '')[:80] or '真实数据导入'
        else:
            resolved,archive,run_name=await asyncio.to_thread(remote_import.fetch_remote_archive,url,base)
            try:
                sampling=remote_import.store_coco_subset(archive,unpack)
            finally:
                try: archive.unlink()
                except OSError: pass
        report=import_coco(unpack/'images',unpack/'coco.json',base/'dataset','real_'+id,url,note or '未登记授权（链接导入未填写，发布前请补齐）')
    except (zipfile.BadZipFile,KeyError,OSError) as exc: raise ValueError('来源内容不是含coco.json和images目录的合法COCO ZIP') from exc
    run=Run(id=id,project_id=project,name=run_name or '真实数据链接导入',relative_path=(base/'dataset').relative_to(ROOT).as_posix(),route='real')
    detail={'count':report['samples'],'source':url,'license':note or '未登记（链接导入）'}
    if sampling: detail['sampling']=sampling
    db.add(run); audit(db,project,user.username,'导入真实数据',**detail); db.commit(); return {**run_view(run),'sampling':sampling}

@app.get('/api/projects/{project}/experiments')
def experiments(project:str,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project)
    return [{k:getattr(e,k) for k in ('id','kind','result','created')} for e in db.scalars(select(Experiment).where(Experiment.project_id==project).order_by(Experiment.created.desc()))]

@app.post('/api/projects/{project}/experiments')
def evaluate(project:str,data:dict,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project,True,{'product','architect','tester'})
    kind=data.get('kind')
    if kind=='physical': result=physical_error(data['reference'],data['prediction'])
    elif kind=='detection': result=detection_metrics(data['truth'],data['predictions'])
    elif kind=='fid':
        real=scoped(db,Run,data['real_run'],user); synthetic=scoped(db,Run,data['synthetic_run'],user)
        if real.project_id!=project or synthetic.project_id!=project or real.id==synthetic.id: raise ValueError('需选择当前项目内不同的真实/合成批次')
        if real.route!='real': raise ValueError('真实参照必须是已登记来源和授权的真实导入数据')
        if min(len(sample_views(real)),len(sample_views(synthetic)))<2: raise ValueError('每组至少2张图；少量样本的FID只能作诊断')
        payload={**data,'real_path':real.relative_path+'/images','synthetic_path':synthetic.relative_path+'/images','idempotency_key':uid()}
        return enqueue(db,project,user,payload,'fid')
    else: raise ValueError('请选择physical、detection或fid评价')
    row=Experiment(id=uid(),project_id=project,kind=kind,result=result); db.add(row); audit(db,project,user.username,'计算评价指标',kind=kind); db.commit()
    return {'id':row.id,'kind':kind,'result':result}

@app.get('/api/projects/{project}/engineering')
def engineering_view(project:str,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project); return engineering(db,project)

@app.get('/api/projects/{project}/events')
def events(project:str,user=Depends(current_user),db=Depends(db_session)):
    access(db,user,project)
    return [{k:getattr(e,k) for k in ('id','actor','action','detail','created')} for e in db.scalars(select(Event).where(Event.project_id==project).order_by(Event.created.desc()).limit(80))]

@app.get('/api/settings')
def settings(user=Depends(current_user),db=Depends(db_session)):
    config=label_config(db); worker=db.get(Setting,'worker_heartbeat')
    connected=bool(worker and time.time()-worker.value.get('at',0)<30)
    return {'database':'PostgreSQL' if engine.dialect.name=='postgresql' else 'SQLite（本地开发）',
            'worker':{'online':connected,'last_seen':worker.value.get('at') if worker else None},
            'label_studio':{'url':config.get('url'),'configured':bool(config.get('token'))},
            'stacks':[{'name':n,'purpose':p,'status':s} for n,p,s in [
                ('FDS','工况与物理场求解','worker' if connected else 'offline'),
                ('Blender','多场景渲染与候选掩膜','worker' if connected else 'offline'),
                ('Stable Diffusion','SD1.5外观增强','worker' if connected else 'offline'),
                ('Label Studio','人工标注与任务同步','ready' if config.get('token') else 'configure'),
                ('DVC','不可变快照与版本导出','ready'),('Web','统一操作与证据追溯','ready')]],
            'metrics_url':'/metrics','docs_url':'/docs',
            'monitoring':{'prometheus':'http://localhost:9090','grafana':'http://localhost:3001'}}

@app.post('/api/settings/label-studio')
def save_label(data:dict,user=Depends(current_user),db=Depends(db_session)):
    if not user or user.role not in ADMIN_ROLES|{'ops'}: raise HTTPException(403,'仅架构师（管理员）或运维师可以配置集成')
    url=str(data.get('url','')).rstrip('/')
    # Service configuration is local-only; public SSRF destinations are not accepted.
    if not re.fullmatch(r'http://(localhost|127\.0\.0\.1|annotation-platform)(:\d{2,5})?',url): raise ValueError('仅支持本机或Compose内Label Studio地址')
    row=db.get(Setting,'label_studio')
    value={'url':url,'token':str(data.get('token','')),'auth_type':data.get('auth_type','Token')}
    if value['auth_type'] not in {'Token','Bearer'}: raise ValueError('认证类型不合法')
    if row: row.value=value
    else: db.add(Setting(key='label_studio',value=value))
    db.commit(); return {'configured':bool(value['token'])}

@app.get('/api/documents')
def documents(path:str):
    allowed={'SRS.md','docs/design/软件设计与低成本实施方案.md','docs/design/10_ER全局联系.png',
             'docs/design/02_系统架构.png','docs/统一火焰链路.md','docs/统一平台使用与部署.md','docs/开源参考与实现映射.md','docs/火焰仿真接入.md','docs/FirePhantom接入与能力核对.md','docs/平台测试计划与报告.md','artifacts/platform/verification.json'}
    if path not in allowed or not (ROOT/path).is_file(): raise HTTPException(404,'文档不存在')
    return FileResponse(ROOT/path)

def worker_auth(request):
    expected=os.getenv('AI07_WORKER_KEY') or (STATE/'worker.key').read_text(encoding='utf-8')
    if not hmac.compare_digest(request.headers.get('x-worker-key',''),expected): raise HTTPException(403,'工作进程认证失败')

@app.post('/api/internal/claim')
def claim(request:Request,db=Depends(db_session)):
    worker_auth(request)
    for interrupted in db.scalars(select(Job).where(Job.status=='running',Job.updated<time.time()-90)):
        interrupted.status='interrupted';interrupted.stage='interrupted';interrupted.log+='\n租约心跳已超时；终止本次记录，需手动重试。'
    heart=db.get(Setting,'worker_heartbeat')
    if heart: heart.value={'at':time.time()}
    else: db.add(Setting(key='worker_heartbeat',value={'at':time.time()}))
    # PostgreSQL workers skip rows leased by other workers; SQLite uses atomic CAS.
    query=select(Job).where(Job.status=='queued',Job.cancel_requested==False).order_by(Job.created).limit(1)
    if engine.dialect.name=='postgresql': query=query.with_for_update(skip_locked=True)
    row=db.scalar(query)
    if not row: db.commit(); return None
    lease=uid(); result=db.execute(update(Job).where(Job.id==row.id,Job.status=='queued').values(status='running',stage='preparing',lease=lease,updated=time.time()))
    if not result.rowcount: db.rollback(); return None
    db.commit(); db.refresh(row); return {**job_view(row),'lease':lease}

@app.post('/api/internal/jobs/{id}')
def worker_update(id:str,data:dict,request:Request,db=Depends(db_session)):
    worker_auth(request); row=db.get(Job,id)
    if not row or not hmac.compare_digest(row.lease,str(data.get('lease',''))) or row.status!='running': raise HTTPException(409,'任务租约已失效')
    row.updated=time.time()
    if 'log' in data: row.log=(row.log+str(data['log']))[-80000:]
    if 'stage' in data: row.stage=str(data['stage'])[:40]
    heart=db.get(Setting,'worker_heartbeat')
    if heart: heart.value={'at':time.time()}
    status=data.get('status')
    if status:
        if status not in {'succeeded','failed','cancelled'}: raise ValueError('非法终态')
        row.status=status; row.stage=status; row.result=data.get('result',{})
        if status=='succeeded' and row.kind=='synthesis':
            from fire_modeling.synthesis_pipeline import validate_synthesis
            counts=validate_synthesis(ROOT/f'artifacts/platform/jobs/{row.id}',row.payload)
            relative=f'artifacts/platform/jobs/{row.id}/'+('enhanced' if row.payload['enhance'] else 'render')
            run=Run(id=row.id,project_id=row.project_id,name=row.payload['name'],relative_path=relative,route='physics')
            db.add(run);row.result={**row.result,'run_id':run.id,'samples':counts['raw']+counts['augmented'],'synthesis_counts':counts}
        elif status=='succeeded' and row.kind in {'generate','editor'}:
            relative=f'artifacts/platform/jobs/{row.id}/'+('enhanced' if row.payload.get('enhance') and row.payload['route']=='physics' else 'render')
            if not (ROOT/relative/'manifest.json').is_file(): raise ValueError('产物清单不存在，禁止标记成功')
            manifest=read_json(ROOT/relative/'manifest.json')
            if manifest.get('status')!='success_candidates_only': raise ValueError('产物未完整生成')
            if row.payload.get('pipeline_schema')=='fds-volume-v1':
                from fire_modeling.volume import validate_volume_run
                validate_volume_run(ROOT/relative,row.payload['config'])
            run=Run(id=row.id,project_id=row.project_id,name=row.payload.get('name') or row.payload['config']['scene_id'],relative_path=relative,route=row.payload['route'])
            db.add(run); row.result={**row.result,'run_id':run.id,'samples':len(manifest['samples'])}
            if manifest.get('rejected_enhancements'):
                row.result['rejected_enhancements']=len(manifest['rejected_enhancements'])
            if (ROOT/f'artifacts/platform/jobs/{row.id}/physics/web/simulation.json').is_file():
                row.result['simulation_url']=f'/api/simulations/{row.id}'
        elif status=='succeeded' and row.kind=='video':
            from fire_modeling.video import validate_video
            video=validate_video(ROOT/f'artifacts/platform/jobs/{row.id}',row.payload['config'])
            row.result={**row.result,'video_url':f'/api/simulation-videos/{row.id}',
                'simulation_url':f'/api/simulations/{row.id}','videos':len(video['videos'])}
            relative=f'artifacts/platform/jobs/{row.id}/video'
            if (ROOT/relative/'manifest.json').is_file():
                manifest=read_json(ROOT/relative/'manifest.json')
                db.add(Run(id=row.id,project_id=row.project_id,name=row.payload['name'],relative_path=relative,route='physics'))
                row.result.update(run_id=row.id,samples=len(manifest['samples']))
        elif status=='succeeded' and row.kind=='fid':
            db.add(Experiment(id=uid(),project_id=row.project_id,kind='fid',result=row.result))
        elif status=='succeeded' and row.kind=='simulation':
            directory=ROOT/f'artifacts/platform/jobs/{row.id}/physics'
            if not (directory/'web/simulation.json').is_file(): raise ValueError('FDS物理场结果不存在，禁止标记成功')
            report=load_simulation(directory)
            execution=read_json(directory/'execution.json')
            if execution.get('status')!='success': raise ValueError('FDS求解未成功')
            if float(execution.get('last_time_s',0))<row.payload['config']['duration_s']-.11:
                raise ValueError('FDS结果未覆盖配置时长')
            for field in report['fields']:
                if not field['frames']: raise ValueError('物理场没有输出时间帧')
                for entry in field['frames']:
                    if sha256(resolve_artifact(directory/'web',entry['file']))!=entry['sha256']: raise ValueError('物理场文件校验失败')
            for source in report.get('sources',[]):
                if sha256(resolve_artifact(directory,source['file']))!=source['sha256']: raise ValueError('原始求解结果校验失败')
            row.result={**row.result,'simulation_url':f'/api/simulations/{row.id}',
                        'fields':[f['id'] for f in report['fields']],'solver_revision':report.get('solver_revision')}
        audit(db,row.project_id,'worker','任务结束',job=row.id,status=status)
    db.commit(); return {'cancel_requested':row.cancel_requested}

@app.get('/api/health')
def health(): return {'status':'ok','uptime_s':round(time.time()-started),'database':engine.dialect.name}

@app.get('/metrics',response_class=PlainTextResponse)
def metrics(db=Depends(db_session)):
    lines=['# TYPE ai07_http_requests_total counter',f'ai07_http_requests_total {REQUESTS["total"]}',
           '# TYPE ai07_http_errors_total counter',f'ai07_http_errors_total {REQUESTS["errors"]}',
           '# TYPE ai07_uptime_seconds gauge',f'ai07_uptime_seconds {time.time()-started:.2f}']
    for status,count in db.execute(select(Job.status,func.count()).group_by(Job.status)):
        lines.append(f'ai07_jobs{{status="{status}"}} {count}')
    return '\n'.join(lines)+'\n'

from .editor_compat import router as editor_compat_router
app.include_router(editor_compat_router)
from .synthesis import router as synthesis_router
app.include_router(synthesis_router)

from .workflow import router as workflow_router
app.include_router(workflow_router)
