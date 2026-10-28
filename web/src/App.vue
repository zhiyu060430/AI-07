<script setup lang="ts">
import {computed,nextTick,onMounted,onUnmounted,ref,watch} from 'vue'
import {Flame,LayoutDashboard,SlidersHorizontal,Images,GitBranch,ChartNoAxesCombined,ShieldCheck,ArrowUpRight,ArrowRight,Plus,Play,Search,RefreshCw,Check,ChevronRight,LogIn,LogOut,Menu,X,Layers,Box,Activity,Cloud,Download,ExternalLink,Clock,Terminal,CheckCircle2,AlertCircle,Filter,Upload,Users,UserPlus,Settings2,RotateCcw,Square,ScanLine,Database,Zap,Eye} from 'lucide-vue-next'
import {api,time,statusNames as sn} from './api'
import BoxEditor from './components/BoxEditor.vue'
import BatchWorkflow from './components/BatchWorkflow.vue'
import EvaluationWorkbench from './components/EvaluationWorkbench.vue'
import JobCenter from './components/JobCenter.vue'
import FlameWorkbench from './components/FlameWorkbench.vue'
import SceneEditorPage from './components/SceneEditorPage.vue'

const nav=[{id:'overview',name:'工作台',icon:LayoutDashboard},{id:'simulation',name:'仿真与合成',icon:Flame},{id:'jobs',name:'任务中心',icon:Activity},{id:'samples',name:'样本与标注',icon:Images},{id:'versions',name:'版本与导出',icon:GitBranch},{id:'evaluation',name:'效果评价',icon:ChartNoAxesCombined},{id:'engineering',name:'工程与验收',icon:ShieldCheck}]
const page=ref(location.hash.slice(1)||'overview'),project=ref(new URLSearchParams(location.search).get('project')||'ai07'),projects=ref<any[]>([]),auth=ref<any>({user:null,setup_required:false})
const overview=ref<any>(null),samples=ref<any[]>([]),jobs=ref<any[]>([]),versions=ref<any[]>([]),experiments=ref<any[]>([]),scenes=ref<any[]>([]),events=ref<any[]>([]),settings=ref<any>(null),engineering=ref<any>(null)
const loading=ref(true),busy=ref(false),error=ref(''),toast=ref(''),sidebarOpen=ref(false),showAuth=ref(false),search=ref(''),source=ref('all'),reviewFilter=ref('all'),qualityFilter=ref('all'),runFilter=ref(new URLSearchParams(location.search).get('run')||'all'),selected=ref<any>(null),selectedJob=ref<any>(null)
const authDialog=ref<HTMLElement|null>(null);let previousFocus:HTMLElement|null=null
watch(showAuth,async value=>{if(value){previousFocus=document.activeElement as HTMLElement;authTab.value=auth.value.setup_required?'register':'login';authError.value='';await nextTick();authDialog.value?.querySelector<HTMLInputElement>('input')?.focus()}else previousFocus?.focus()})
function handleDialogKey(event:KeyboardEvent){if(!showAuth.value)return;if(event.key==='Escape'){event.preventDefault();showAuth.value=false;return}if(event.key==='Tab'){const controls=[...(authDialog.value?.querySelectorAll<HTMLElement>('button:not([disabled]),input,select,a[href]')||[])];const first=controls[0],last=controls.at(-1);if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus()}else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus()}}}
const loginForm=ref({username:'',password:''}),authError=ref(''),newUser=ref({username:'',password:'',role:'developer'}),members=ref<any[]>([]),membersLoading=ref(false),membersError=ref('')
const canManageUsers=computed(()=>auth.value.user?.role==='architect')
const visibleNav=computed(()=>canManageUsers.value?[...nav,{id:'members',name:'项目成员',icon:Users}]:nav)
const authTab=ref('login'),registerForm=ref({username:'',password:'',role:'tester'})
const config=ref<any>({scene_id:'room_demo',scene_type:'indoor',room_size_m:[4,4,3],cell_size_m:.2,fire_center_m:[2,2],fire_size_m:[.8,.8],peak_hrr_kw:100,duration_s:5,ramp_up_s:2,ambient_temperature_c:20,wind_m_s:0,seed:42,width:384,height:384,camera_angles_deg:[25,65],weather:['clear','fog'],plume_steps:45,particles_per_step:10})
const generate=ref({name:'室内火焰 · 多视角',route:'physics',count:2,enhance:true}),versionForm=ref({run_id:new URLSearchParams(location.search).get('run')||'',name:'v0.1.0',research_snapshot:false}),labelForm=ref({url:'http://127.0.0.1:8080',token:'',auth_type:'Token'})
const continuousReview=ref(false),labelState=ref<any>(null),labelBusy=ref(false),labelError=ref('')
const uploadFile=ref<File|null>(null),sourceUrl=ref(''),license=ref(''),showImport=ref(false)
const titles:Record<string,string>={members:'项目成员','scene-editor':'三维场景编辑',simulation:'火焰仿真与数据合成',overview:'火焰数据工作台',generate:'从场景开始，生成可追溯的数据',jobs:'任务中心',samples:'样本与标注',versions:'版本与导出',evaluation:'用真实证据，评价合成数据',engineering:'工程与验收'}
const kickers:Record<string,string>={simulation:'SIMULATION & SYNTHESIS',jobs:'OPERATIONS',samples:'SAMPLE DATA',versions:'VERSION CONTROL',evaluation:'EVIDENCE LAB',engineering:'ENGINEERING','scene-editor':'3D STUDIO'}
const subtitles:Record<string,string>={members:'为当前项目创建成员账户，设置初始密码并分配角色。','scene-editor':'模型、燃烧物、结构、灯光与相机，共用一份场景进入FDS与Blender。',simulation:'三维编辑页完成仿真；在这里复用求解结果，生成并管理数据批次。',overview:'物理仿真、生成增强与人工审核，在同一个工作区协作。',generate:'配置工况、选择路线，参数会随每次执行一起保留。',jobs:'查看每个阶段的进度、日志和产物。',samples:'候选标签只是起点。每个发布样本，都需要人工确认。',versions:'冻结图像、标签与审核记录，使用DVC和Git保留每一次变化。',evaluation:'分布相似度、物理误差与检测收益，各自使用独立评价依据。',engineering:'把功能、指标和统一交付要求，落实到可检查的证据。'}
const filtered=computed(()=>samples.value.filter(s=>(!search.value||`${s.sample_id} ${s.run_name}`.toLowerCase().includes(search.value.toLowerCase()))&&(source.value==='all'||s.route===source.value)&&(reviewFilter.value==='all'||s.review_status===reviewFilter.value)&&(runFilter.value==='all'||s.run_id===runFilter.value)&&(qualityFilter.value==='all'||(qualityFilter.value==='pass'?s.quality?.auto_screen_pass===true:s.quality?.auto_screen_pass===false))))
const runs=computed(()=>overview.value?.runs||[])
const roleNames:Record<string,string>={product:'产品经理',architect:'架构师（管理员）',developer:'开发工程师',tester:'测试师',ops:'运维师'}
const featured=computed(()=>[...samples.value].sort((a,b)=>Number(b.expected_fire)-Number(a.expected_fire)||Number(a.route==='preview')-Number(b.route==='preview')).slice(0,4))
const stageNames:Record<string,string>={queued:'等待工作进程',preparing:'检查参数','synthesis-render':'多条件基础渲染','synthesis-enhance':'受控扩散增强',original:'原版Mantaflow烘焙与渲染',video:'固定机位渲染与MP4编码',physics:'FDS物理求解',blender:'Blender批量渲染',enhance:'SD生成增强',preview:'程序预览',fid:'Inception/FID评价',succeeded:'产物生成完成',failed:'阶段执行失败',cancelled:'任务已取消'}
const curvePath=computed(()=>{const c=overview.value?.curve||[];if(!c.length)return ''; const maxX=Math.max(...c.map((p:any)=>p.time),1),maxY=Math.max(...c.map((p:any)=>p.hrr),1);return c.map((p:any,i:number)=>`${i?'L':'M'}${36+p.time/maxX*514},${150-p.hrr/maxY*116}`).join(' ')})
const curveAreaPath=computed(()=>curvePath.value?curvePath.value+' L550,150 L36,150 Z':'')
function eventIcon(action:string){const a=action||'';if(a.includes('注册'))return UserPlus;if(a.includes('结束')||a.includes('完成')||a.includes('成功'))return CheckCircle2;if(a.includes('失败')||a.includes('中断')||a.includes('取消'))return AlertCircle;if(a.includes('审核')||a.includes('质检')||a.includes('标注'))return ScanLine;if(a.includes('发布')||a.includes('版本')||a.includes('快照'))return GitBranch;if(a.includes('任务'))return Play;return Activity}
const revealObserver=window.matchMedia('(prefers-reduced-motion: reduce)').matches?null:new IntersectionObserver(entries=>{for(const e of entries){if(e.isIntersecting){e.target.classList.add('in-view');revealObserver?.unobserve(e.target)}}},{threshold:.12})
const vReveal={mounted(el:HTMLElement){if(!revealObserver){el.classList.add('reveal','in-view');return}el.classList.add('reveal');revealObserver.observe(el)},unmounted(el:HTMLElement){revealObserver?.unobserve(el)}}
function fmtNum(n:number){return n.toLocaleString('en-US')}
function animNum(el:HTMLElement,from:number,to:number){const reduce=window.matchMedia('(prefers-reduced-motion: reduce)').matches;if(reduce||!isFinite(to)){el.textContent=fmtNum(to);return}const dur=900,t0=performance.now();const tick=(t:number)=>{const k=Math.min(Math.max((t-t0)/dur,0),1);el.textContent=fmtNum(Math.round(from+(to-from)*(1-Math.pow(1-k,3))));if(k<1)requestAnimationFrame(tick)};requestAnimationFrame(tick)}
const vCountup={mounted(el:HTMLElement,binding:{value:number}){el.dataset.v=String(binding.value??0);animNum(el,0,binding.value??0)},updated(el:HTMLElement,binding:{value:number,oldValue:number}){if(binding.value===binding.oldValue)return;const from=Number((el.textContent||'0').replace(/,/g,''))||0;el.dataset.v=String(binding.value??0);animNum(el,from,binding.value??0)}}
let timer:ReturnType<typeof setInterval>|undefined,toastTimer:ReturnType<typeof setTimeout>|undefined
function go(id:string){if(id==='generate')id='simulation';page.value=id;location.hash=id;sidebarOpen.value=false;selected.value=null;error.value='';window.scrollTo({top:0,behavior:'instant'})}
function notify(message:string){toast.value=message;clearTimeout(toastTimer);toastTimer=setTimeout(()=>toast.value='',4500)}
function ensureWrite(){if(!auth.value.user){showAuth.value=true;return false}return true}
async function refresh(quiet=false){
 if(auth.value.require_login&&!auth.value.user){loading.value=false;return}
 if(!quiet)loading.value=true
 try{const p=project.value;const [o,s,j,v,e,t,g,a,c]=await Promise.all([api(`/projects/${p}/overview`),api(`/projects/${p}/samples`),api(`/projects/${p}/jobs`),api(`/projects/${p}/versions`),api(`/projects/${p}/experiments`),api('/settings'),api(`/projects/${p}/engineering`),api(`/projects/${p}/events`),api(`/projects/${p}/scenes`)])
 overview.value=o;samples.value=s;jobs.value=j;versions.value=v;experiments.value=e;settings.value=t;engineering.value=g;events.value=a;scenes.value=c
 if(!versionForm.value.run_id&&o.runs.length)versionForm.value.run_id=o.runs[0].id
 if(selectedJob.value)selectedJob.value=j.find((job:any)=>job.id===selectedJob.value.id)||selectedJob.value
 else selectedJob.value=j.find((job:any)=>job.id===new URLSearchParams(location.search).get('job'))||null
 if(!quiet)error.value=''
 }catch(e){if(!quiet)error.value=(e as Error).message}finally{loading.value=false}
}
async function act(action:()=>Promise<any>,success='操作已完成'){
 if(!ensureWrite())return
 busy.value=true;error.value=''
 try{const result=await action();notify(success);await refresh(true);return result}catch(e){error.value=(e as Error).message}finally{busy.value=false}
}
async function authenticate(){busy.value=true;authError.value='';try{const user=await api('/login','POST',loginForm.value);auth.value={...auth.value,user,setup_required:false};showAuth.value=false;projects.value=await api('/projects');await refresh();notify('已进入工作区')}catch(e){authError.value=(e as Error).message}finally{busy.value=false}}
async function register(){busy.value=true;authError.value='';try{const user=await api(auth.value.setup_required?'/setup':'/register','POST',registerForm.value);auth.value={...auth.value,user,setup_required:false};showAuth.value=false;projects.value=await api('/projects');await refresh();notify('账户已创建，已进入工作区')}catch(e){authError.value=(e as Error).message}finally{busy.value=false}}
async function logout(){await api('/logout','POST');auth.value=await api('/auth');project.value='ai07';if(auth.value.require_login){projects.value=[];overview.value=null;samples.value=[];jobs.value=[];versions.value=[];experiments.value=[];scenes.value=[];events.value=[];settings.value=null;engineering.value=null;selected.value=null;selectedJob.value=null;members.value=[];membersError.value='';showAuth.value=true;return}projects.value=await api('/projects');await refresh()}
function openSample(s:any){runFilter.value=s.run_id;selected.value=s;window.scrollTo({top:0,behavior:'smooth'})}
function selectScene(id:string){const s=scenes.value.find(x=>x.id===id);if(s){config.value=JSON.parse(JSON.stringify(s.config));generate.value.name=s.name+' · 多视角'}}
async function createJob(){const result=await act(()=>api(`/projects/${project.value}/jobs`,'POST',{...generate.value,config:config.value,idempotency_key:crypto.randomUUID()}),'任务已排队，执行状态可在任务中心查看');if(result){selectedJob.value=result;go('jobs')}}
async function loadLabel(){if(runFilter.value==='all'){labelState.value=null;return}try{labelState.value=await api(`/runs/${runFilter.value}/label-studio`)}catch(e){labelError.value=(e as Error).message}}
watch(runFilter,()=>{labelError.value='';loadLabel()},{immediate:true})
function reviewNext(){if(runFilter.value==='all'){error.value='请先选择一个批次';return}continuousReview.value=true;const next=samples.value.find(s=>s.run_id===runFilter.value&&s.review_status==='pending');if(next)openSample(next);else notify('本批次没有待审核样本')}
async function syncLabels(){if(labelBusy.value||!ensureWrite())return;labelBusy.value=true;labelError.value='';try{labelState.value=await api(`/runs/${runFilter.value}/label-studio/push`,'POST')}catch(e){labelError.value='审核已保存在平台。'+(e as Error).message}finally{labelBusy.value=false;await loadLabel()}}
async function saveReview(boxes:number[][],decision:string,timing?:{active_s:number;mode:string}){const s=selected.value;const result=await act(()=>api(`/runs/${s.run_id}/review`,'POST',{sample_id:s.sample_id,boxes,decision,...(timing?{timing}:{})}),'审核记录已保存');if(!result)return;labelState.value=result.label_studio;if(continuousReview.value){const next=samples.value.find(x=>x.run_id===s.run_id&&x.review_status==='pending');selected.value=next||null;if(!next)notify('本批待审核样本已处理完成')}else selected.value=samples.value.find(x=>x.run_id===s.run_id&&x.sample_id===s.sample_id);if(result.label_studio?.connected)void syncLabels()}
async function labelAction(action:string){await act(()=>api(`/runs/${runFilter.value}/label-studio/${action}`,'POST'),action==='pull'?'人工标签已导入':'候选任务已同步');await loadLabel()}
async function evaluationAction(path:string,data:any){return await act(()=>api(path,'POST',data),'评价作业已提交，结果完成后自动显示')}
async function createVersion(){await act(()=>api(`/projects/${project.value}/versions`,'POST',versionForm.value),'DVC研究快照已发布')}
async function upload(){if(!uploadFile.value&&!sourceUrl.value.trim()){error.value='请上传COCO ZIP或填写可下载的数据集链接（二选一）';return}const form=new FormData();if(uploadFile.value)form.append('file',uploadFile.value);form.append('source_url',sourceUrl.value);form.append('license_note',license.value);const result=await act(()=>api(`/projects/${project.value}/import-coco`,'POST',form),'真实数据已导入，仍需人工确认标签');if(result){showImport.value=false;uploadFile.value=null}}
async function loadMembers(){
 members.value=[];membersError.value=''
 if(!canManageUsers.value)return
 const p=project.value;membersLoading.value=true
 try{const result=await api(`/projects/${p}/users`);if(p===project.value&&canManageUsers.value)members.value=result}catch(e){if(p===project.value)membersError.value=(e as Error).message}finally{if(p===project.value)membersLoading.value=false}
}
function manageUsers(){if(!ensureWrite())return;go('members')}
async function addUser(){
 if(busy.value||!ensureWrite())return
 busy.value=true;membersError.value=''
 try{await api(`/projects/${project.value}/users`,'POST',newUser.value);newUser.value={username:'',password:'',role:'developer'};notify('项目成员已创建，可使用初始密码登录');await loadMembers()}catch(e){membersError.value=(e as Error).message}finally{busy.value=false}
}
watch([page,project,()=>auth.value.user?.id],()=>{if(page.value==='members')loadMembers()},{immediate:true})
onMounted(async()=>{window.addEventListener('keydown',handleDialogKey);try{auth.value=await api('/auth');if(auth.value.require_login&&!auth.value.user){loading.value=false;showAuth.value=true}else{projects.value=await api('/projects');await refresh()}}catch(e){error.value=(e as Error).message;loading.value=false}timer=setInterval(()=>{if(!busy.value&&!showAuth.value)refresh(true)},7000)})
onUnmounted(()=>{window.removeEventListener('keydown',handleDialogKey);clearInterval(timer);clearTimeout(toastTimer)})
</script>

<template>
 <div class="app-shell">
  <aside class="sidebar" :class="{open:sidebarOpen}">
   <a class="brand" href="#overview" @click.prevent="go('overview')"><span class="brand-mark"><Flame :size="25" :stroke-width="1.7"/></span><span>焰序<span class="brand-en">FLAME WORKBENCH</span></span></a>
   <div class="workspace-label">AI-07 · 合成火焰数据集</div>
   <nav aria-label="主导航"><button v-for="n in visibleNav" :key="n.id" :class="{selected:page===n.id}" @click="go(n.id)"><component :is="n.icon" :size="18"/><span>{{n.name}}</span><ChevronRight v-if="page===n.id" :size="14"/></button></nav>
   <div class="sidebar-bottom"><div class="local-state"><span class="status-dot" :class="{online:settings?.worker.online}"></span>{{auth.require_login&&!auth.user?'登录后查看工作进程状态':settings?.worker.online?'工作进程已连接':'工作进程未连接'}}</div><p>让数据有来路，让结果可复现。</p></div>
  </aside>
  <div v-if="sidebarOpen" class="sidebar-scrim" @click="sidebarOpen=false"></div>
  <main class="main-shell">
   <header class="topbar"><div class="topbar-left"><button class="icon-button mobile-menu" aria-label="打开导航" @click="sidebarOpen=true"><Menu :size="20"/></button><span class="breadcrumb">工作空间</span><ChevronRight :size="14"/><select aria-label="选择项目" v-model="project" @change="refresh()"><option v-for="p in projects" :value="p.id" :key="p.id">{{p.name}}</option></select><span class="local-badge">本地研究环境</span></div><div class="topbar-right"><button class="icon-button refresh-button" @click="refresh()" aria-label="刷新数据"><RefreshCw :size="17" :class="{spinning:loading}"/></button><template v-if="auth.user"><span class="user-avatar">{{auth.user.username.slice(0,1).toUpperCase()}}</span><span class="user-label">{{auth.user.username}}<small>{{roleNames[auth.user.role]}}</small></span><button class="icon-button" aria-label="退出登录" @click="logout"><LogOut :size="16"/></button></template><button v-else class="text-button" @click="showAuth=true"><LogIn :size="16"/>{{auth.setup_required?'创建工作区账户':'登录工作区'}}</button></div></header>
   <div class="page-content">
    <div v-if="page!=='overview'" class="page-heading"><div><span class="kicker">{{kickers[page]||'WORKBENCH'}}</span><h1>{{titles[page]||titles.overview}}</h1><p>{{subtitles[page]||subtitles.overview}}</p></div><div class="page-heading-actions"><button v-if="page==='jobs'" class="primary" @click="go('generate')"><Plus :size="17"/>新建生成任务</button><button v-if="page==='samples'" class="secondary" @click="showImport=!showImport"><Upload :size="16"/>导入真实数据</button></div></div>
    <div v-if="error" class="message error" role="alert"><AlertCircle :size="18"/><span>{{error}}</span><button class="icon-button" @click="error=''" aria-label="关闭提示"><X :size="16"/></button></div>
    <div v-if="loading&&!overview" class="loading-state"><div v-for="i in 4" :key="i" class="skeleton"></div><p>正在读取工作区与产物清单…</p></div>

    <div class="page-view" :key="page">
    <template v-if="page==='overview'&&overview">
     <section class="hero-block">
      <span class="kicker">AI-07 · 合成火焰数据工作台</span>
      <h1 class="display-xl">让每张合成火焰<br><em class="grad-text">有来路，可验证</em></h1>
      <p class="hero-sub">从三维场景与 FDS 物理仿真，到扩散增强、人工审核与 DVC 版本冻结——每一步都留下可追溯的证据。</p>
      <div class="hero-actions"><button class="primary" @click="go('generate')"><Plus :size="17"/>新建生成任务</button><button class="secondary" @click="go('samples')">查看样本与审核</button></div>
     </section>
     <div class="overview-strip reveal-stagger" v-reveal><div><span>候选样本</span><strong><span v-countup="overview.samples">{{overview.samples}}</span><small>张</small></strong></div><div><span>待人工审核</span><strong><span v-countup="overview.pending">{{overview.pending}}</span><small>张</small></strong></div><div><span>已审核通过</span><strong><span v-countup="overview.approved">{{overview.approved}}</span><small>张</small></strong></div><div><span>生成式增强</span><strong><span v-countup="overview.enhanced">{{overview.enhanced}}</span><small>张</small></strong></div><div><span>真实参照数据</span><strong>{{overview.real||'—'}}<small>{{overview.real?'张':'待导入'}}</small></strong></div></div>
     <section class="pipeline-section" v-reveal><div class="section-heading"><div><h2>一条可追溯的数据链路</h2><p>六项技术，在同一工作区串联——点击任意环节直达。</p></div><span class="quiet-label">AI + 软件工程</span></div><div class="pipeline"><button v-for="(s,i) in settings?.stacks||[]" :key="s.name" @click="go(i===0?'simulation':i<3?'generate':i===3?'samples':i===4?'versions':'engineering')"><span class="pipeline-num">{{String(i+1).padStart(2,'0')}}</span><span class="pipeline-glyph"><component :is="[Activity,Box,Zap,ScanLine,GitBranch,Layers][i]" :size="22"/></span><strong>{{s.name}}</strong><small>{{s.purpose}}</small></button></div></section>
     <div class="section-heading gallery-heading" v-reveal><div><h2>从产物开始检查</h2><p>实际生成结果 · 未审核候选，不作为真实火灾数据。</p></div><button class="text-button" @click="go('samples')">查看全部样本 <ArrowUpRight :size="16"/></button></div>
     <div class="sample-gallery overview-gallery reveal-stagger" v-reveal><article v-for="s in featured" :key="s.run_id+s.sample_id" class="sample-tile"><button class="sample-visual" @click="go('samples');selected=s"><img :src="s.url" :alt="s.sample_id" loading="lazy"><span class="image-badge">{{s.source==='diffusion'?'SD增强':sn[s.route]}}</span><span class="image-action"><Eye :size="16"/>检查样本</span></button><div class="sample-caption"><div><strong>{{s.parent_id?'增强候选':s.expected_fire?'有火候选':'无火候选'}}</strong><span>{{s.width}} × {{s.height}} · {{s.weather==='fog'?'雾天':'晴天'}}</span></div><span class="badge" :class="s.review_status">{{sn[s.review_status]}}</span></div><div class="sample-id">{{s.sample_id}}</div></article></div>
     <div class="overview-bottom" v-reveal><section class="activity-panel"><div class="section-heading"><h2>最近活动</h2><button class="text-button" @click="go('jobs')">任务中心 <ArrowUpRight :size="15"/></button></div><div v-if="!events.length" class="first-step"><span class="line-icon"><Play :size="22"/></span><div><h3>下一步，创建第一项平台任务</h3><p>现有实跑产物已登记。新任务会记录执行阶段、日志和审核轨迹。</p></div><button class="text-button" @click="go('generate')">开始配置 <ArrowRight :size="16"/></button></div><div v-for="e in events.slice(0,4)" :key="e.id" class="activity-row"><span class="activity-ico"><component :is="eventIcon(e.action)" :size="13"/></span><div><strong>{{e.action}}</strong><small>{{e.actor}}</small></div><time>{{time(e.created)}}</time></div></section><section class="curve-panel"><div class="section-heading"><h2>物理求解观察</h2><span class="badge neutral">FDS · HRR</span></div><svg v-if="curvePath" class="curve-chart" viewBox="0 0 580 178" role="img" aria-label="实际FDS热释放率随时间变化"><defs><linearGradient id="hrrArea" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#f08a3c" stop-opacity=".28"/><stop offset="1" stop-color="#f08a3c" stop-opacity="0"/></linearGradient></defs><path class="chart-grid" d="M36 40H550 M36 95H550 M36 150H550"/><text x="6" y="24">kW</text><text x="36" y="172">0 s</text><text x="520" y="172">{{overview.curve.at(-1)?.time.toFixed(1)}} s</text><path :d="curveAreaPath" class="chart-area" fill="url(#hrrArea)"/><path :d="curvePath" class="chart-line"/></svg><p v-else class="empty-inline">生成物理批次后，在这里查看真实HRR曲线。</p><p class="context-note">求解曲线不等于独立物理精度验证。</p></section></div>
     <div class="workspace-footer"><span><ShieldCheck :size="15"/>候选 → 人工审核 → DVC快照 → 独立评价</span><button class="text-button" @click="go('engineering')">查看工程与验收 <ArrowRight :size="15"/></button></div>
    </template>

    <template v-else-if="page==='scene-editor'"><SceneEditorPage :project="project"/></template>
    <template v-else-if="page==='simulation'||page==='generate'">
     <FlameWorkbench :key="project" :project="project" :config="config" :scenes="scenes" :submit="data=>act(()=>api(`/projects/${project}/jobs`,'POST',data),'仿真与合成任务已排队')" :simulate="data=>act(()=>api(`/projects/${project}/simulations`,'POST',data),'FDS试验已排队')" @navigate="(target,run)=>{go(target);if(run){runFilter=run;versionForm.run_id=run}}"/>
    </template>

    <template v-else-if="page==='jobs'">
     <JobCenter :jobs="jobs" :selected="selectedJob" :busy="busy" :project="project" @select="selectedJob=$event" @cancel="j=>act(()=>api(`/jobs/${j.id}/cancel`,'POST'),'取消请求已提交')" @retry="async j=>{const result=await act(()=>api(`/jobs/${j.id}/retry`,'POST'),'已创建重试任务');if(result)selectedJob=result}" @samples="run=>{go('samples');runFilter=run}"/>
    </template>

    <template v-else-if="page==='samples'">
     <BatchWorkflow :runs="runs" :run-id="runFilter" :scenes="scenes" :samples="samples" @navigate="id=>{if(id==='versions'&&runFilter!=='all')versionForm.run_id=runFilter;go(id)}" @review="reviewNext"/>
     <section v-if="showImport" class="import-panel"><div class="section-heading"><div><h2>导入有来源的真实数据</h2><p>二选一：上传 COCO ZIP（coco.json+images，类别ID为1 fire，≤50MB，需另填来源与授权），或直接粘贴数据集链接——figshare 页面链接可直接粘贴，平台自动解析下载并抽样解包（远端≤800MB，导入≤3000条/150MB）。</p></div><button class="icon-button" @click="showImport=false" aria-label="关闭导入"><X :size="18"/></button></div><form class="form-grid import-form" @submit.prevent="upload"><label>COCO ZIP（留空则从链接下载）<input type="file" accept=".zip" @change="uploadFile=($event.target as HTMLInputElement).files?.[0]||null"></label><label>原始来源链接<input v-model="sourceUrl" type="url" required placeholder="数据集页面或ZIP直链 https://…"></label><label>授权或许可证说明<input v-model="license" placeholder="上传ZIP必填；链接导入可留空待补"></label><button class="primary" :disabled="busy"><Upload :size="16"/>{{uploadFile?'导入并登记来源':'从链接下载并导入'}}</button></form></section>
     <BoxEditor v-if="selected" :sample="selected" :busy="busy" :continuous="continuousReview" :parent="samples.find(s=>s.run_id===selected.run_id&&s.sample_id===selected.parent_id)" @save="saveReview" @close="selected=null;continuousReview=false"/>
     <div class="filter-bar"><label class="search-field"><Search :size="17"/><input v-model="search" aria-label="搜索样本" placeholder="搜索样本或批次"></label><select v-model="runFilter" aria-label="批次筛选"><option value="all">全部生成批次</option><option v-for="r in runs" :key="r.id" :value="r.id">{{r.name}}</option></select><select v-model="source" aria-label="来源筛选"><option value="all">全部来源</option><option value="physics">物理生成</option><option value="preview">程序预览</option><option value="real">真实导入</option></select><select v-model="reviewFilter" aria-label="审核筛选"><option value="all">全部审核状态</option><option value="pending">待审核</option><option value="approved">已通过</option><option value="rejected">已剔除</option></select><select v-model="qualityFilter" aria-label="质量筛选"><option value="all">全部质量状态</option><option value="pass">自动筛选通过</option><option value="issue">需检查质量</option></select><span class="result-count">{{filtered.length}}张样本</span></div>
     <div v-if="runFilter!=='all'" class="run-toolbar"><span><ScanLine :size="17"/>批次操作</span><button class="text-button" :disabled="busy" @click="act(()=>api(`/runs/${runFilter}/quality`,'POST'),'结构质检已重新计算')">重新质检</button><button class="text-button" :disabled="busy" @click="reviewNext">页内连续审核</button><button class="text-button" :disabled="labelBusy" @click="syncLabels">{{labelBusy?'后台同步中…':labelState?.connected?'重试/同步Label Studio':'连接并同步Label Studio'}}</button><button class="text-button" :disabled="busy||labelBusy||!labelState?.connected" @click="labelAction('pull')">回导外部人工标签</button></div>
     <p v-if="runFilter!=='all'" class="context-note">直接在本页框选、修订并审核，只需平台登录。{{labelState?.connected?'Label Studio已连接；待同步 '+labelState.pending_sync+' 条审核记录。':'连接后，页内审核将通过后台接口同步到Label Studio。'}}<a v-if="labelState?.url" :href="labelState.url" target="_blank" rel="noreferrer">完整工具（可选）</a><span v-if="labelState?.connected">，完整工具需用 Label Studio 服务账号单独登录（账号见部署文档）</span></p><p v-if="labelError" class="message error" role="alert">{{labelError}}</p>
     <div class="sample-gallery"><article class="sample-tile" v-for="s in filtered" :key="s.run_id+s.sample_id"><button class="sample-visual" @click="openSample(s)"><img :src="s.url" :alt="s.sample_id" loading="lazy"><svg :viewBox="`0 0 ${s.width} ${s.height}`" class="candidate-overlay" aria-hidden="true"><rect v-for="(b,i) in s.boxes" :key="i" :x="b[0]" :y="b[1]" :width="b[2]-b[0]" :height="b[3]-b[1]"/></svg><span class="image-badge">{{s.source==='diffusion'?'SD增强':sn[s.route]}}</span><span class="image-action"><ScanLine :size="16"/>检查与编辑</span></button><div class="sample-caption"><div><strong>{{s.sample_id}}</strong><span>{{s.width}} × {{s.height}} · {{s.boxes.length}}个候选框</span></div><span class="badge" :class="s.review_status">{{sn[s.review_status]}}</span></div><div class="sample-id"><span class="quality-dot" :class="{pass:s.quality?.auto_screen_pass}"></span>{{s.quality?(s.quality.auto_screen_pass===false?'需检查质量或候选框':s.quality.structural_pass?'结构质检通过':'结构质检有问题'):'尚无质检记录'}}</div></article></div><div v-if="!filtered.length" class="empty-state"><Images :size="36"/><h2>没有符合筛选条件的样本</h2><p>调整筛选条件，或先创建生成任务、导入真实数据。</p></div>
    </template>

    <template v-else-if="page==='versions'">
     <div class="version-layout"><section class="form-panel"><div class="section-heading"><h2>创建研究快照</h2><GitBranch :size="20"/></div><form @submit.prevent="createVersion" class="stacked-form"><label>来源批次<select v-model="versionForm.run_id" required><option value="" disabled>选择已审核批次</option><option v-for="r in runs" :key="r.id" :value="r.id">{{r.name}} · 已通过{{r.approved}}/{{r.count}}</option></select></label><label>版本名称<input v-model="versionForm.name" required pattern="[A-Za-z0-9][A-Za-z0-9_.-]{0,63}"></label><label class="checkbox-label"><input v-model="versionForm.research_snapshot" type="checkbox" required>确认这是研究快照，不代表课程正式验收通过</label><p class="context-note">只纳入人工审核通过的样本。图像、标签和分组被冻结；旧版本不能覆盖。少量源组可能导致部分集合为空。</p><button class="primary" :disabled="busy"><GitBranch :size="16"/>{{busy?'DVC提交与备份中…':'冻结并发布DVC快照'}}</button></form><ol class="compact-steps"><li>检查人工审核、哈希、结构与重复</li><li>按源组划分训练/验证/测试集合</li><li>DVC add → Git commit → DVC push</li><li>核验本地备份后开放COCO / YOLO下载</li></ol></section><section class="version-history"><div class="section-heading"><h2>版本记录</h2><span class="quiet-label">{{versions.length}}个快照</span></div><div v-if="!versions.length" class="empty-state"><GitBranch :size="36"/><h3>先审核，再发布</h3><p>没有已审核数据时，平台会阻止发布。</p><button class="text-button" @click="go('samples')">去审核样本 <ArrowRight :size="16"/></button></div><article v-for="v in versions" :key="v.id" class="version-row"><div class="section-heading"><h3>{{v.name}}</h3><span class="badge" :class="v.status">{{sn[v.status]}}</span></div><p v-if="v.status==='published'">{{v.result.count}}张已审核样本 · COCO / YOLO</p><p v-if="v.result.commit" class="commit-code">Git {{v.result.commit.slice(0,12)}} · {{time(v.created)}}</p><p v-if="v.result.splits" class="context-note">训练{{v.result.splits.train}} / 验证{{v.result.splits.val}} / 测试{{v.result.splits.test}} · 同源组不跨集合</p><p class="context-note">{{v.result.backup||v.result.error}}</p><a v-if="v.status==='published'" class="secondary small" :href="`/api/versions/${v.id}/download`"><Download :size="15"/>下载数据集</a></article></section></div>
    </template>

    <template v-else-if="page==='evaluation'">
     <EvaluationWorkbench :key="project" :project="project" :runs="runs" :versions="versions" :experiments="experiments" :jobs="jobs" :busy="busy" :execute="evaluationAction" @import="go('samples');showImport=true" @review="go('samples')" @job="id=>{selectedJob=jobs.find(j=>j.id===id);go('jobs')}"/>
    </template>

    <template v-else-if="page==='members'">
     <section v-if="!auth.user" class="form-panel"><p>请先登录管理员账户，再管理项目成员。</p><button class="primary" @click="showAuth=true">登录工作区</button></section>
     <section v-else-if="!canManageUsers" class="form-panel"><h2>需要管理员权限</h2><p>当前角色为{{roleNames[auth.user.role]}}。仅管理员可以创建项目成员，请联系项目管理员。</p></section>
     <section v-else class="form-panel user-management" aria-label="项目成员管理" :aria-busy="membersLoading">
      <div class="section-heading"><h2>当前项目成员</h2><button class="text-button" :disabled="membersLoading||busy" @click="loadMembers"><RefreshCw :size="15"/>刷新成员</button></div>
      <div v-if="membersError" class="message error" role="alert">{{membersError}}</div>
      <p v-if="membersLoading" role="status">正在读取项目成员…</p>
      <div v-else class="member-list"><span v-for="m in members" :key="m.id">{{m.username}} <small>{{roleNames[m.role]}}</small></span><p v-if="!members.length&&!membersError">当前项目暂无成员。</p></div>
      <h2>添加项目成员</h2><p class="context-note">创建后，成员可以用自己的账户登录同一平台。管理员负责创建和管理成员；其他角色按分配的权限操作项目。</p>
      <form class="form-grid" @submit.prevent="addUser"><label>用户名<input v-model="newUser.username" required minlength="3" maxlength="32" pattern="[A-Za-z0-9_-]{3,32}" autocomplete="off" aria-describedby="member-username-help"><small id="member-username-help">3–32位字母、数字、下划线或短横线</small></label><label>初始密码<input v-model="newUser.password" type="password" required minlength="10" autocomplete="new-password"><small>至少10位，通过私密渠道告知成员</small></label><label>角色<select v-model="newUser.role"><option v-for="(name,id) in roleNames" :key="id" :value="id">{{name}}</option></select></label><button class="primary" :disabled="busy||membersLoading">{{busy?'正在添加…':'添加项目成员'}}</button></form>
     </section>
    </template>

    <template v-else-if="page==='engineering'&&engineering">
     <div class="engineering-top" v-reveal><section class="runtime-panel"><div class="section-heading"><div><span class="sec-kicker">RUNTIME</span><h2>运行与集成</h2></div><span class="badge neutral">{{settings?.database}}</span></div><div class="integration-list"><div v-for="s in settings?.stacks||[]" :key="s.name"><span class="integration-glyph"><component :is="[Activity,Box,Zap,ScanLine,GitBranch,Layers][settings?.stacks.indexOf(s)]" :size="17"/></span><strong>{{s.name}}</strong><span>{{s.purpose}}</span><span class="badge" :class="s.status==='ready'||s.status==='worker'?'approved':'pending'">{{s.status==='ready'?'已接入':s.status==='worker'?'Worker在线':s.status==='configure'?'需配置令牌':'Worker离线'}}</span></div></div><div class="button-row external-tools"><a href="/docs" target="_blank" class="text-button">OpenAPI <ArrowUpRight :size="14"/></a><a href="http://localhost:9090" target="_blank" class="text-button">Prometheus <ArrowUpRight :size="14"/></a><a href="http://localhost:3001" target="_blank" class="text-button">Grafana <ArrowUpRight :size="14"/></a><a href="http://localhost:9000" target="_blank" class="text-button">SonarQube <ArrowUpRight :size="14"/></a></div><p class="context-note">监控和Sonar按Compose配置启动；链接存在不代表服务已启动或门禁已通过。</p></section><section class="form-panel"><div class="section-heading"><div><span class="sec-kicker">ANNOTATION</span><h2>Label Studio连接</h2></div><Settings2 :size="18"/></div><form class="stacked-form" @submit.prevent="act(()=>api('/settings/label-studio','POST',labelForm),'集成配置已保存')"><label>本机服务地址<input v-model="labelForm.url" placeholder="http://127.0.0.1:8080" required></label><div class="form-grid"><label>令牌类型<select v-model="labelForm.auth_type"><option value="Token">Legacy Token</option><option value="Bearer">Bearer Token</option></select></label><label>服务令牌<input type="password" v-model="labelForm.token" autocomplete="off" required placeholder="仅保存到后端"></label></div><button class="secondary" :disabled="busy">保存连接设置</button><p class="context-note">已配置：{{settings?.label_studio.configured?'是':'否'}}。在样本页选择批次后同步候选任务，人工标注完成后再导入。</p></form></section></div>
     <section class="requirements-section" v-reveal><div class="section-heading"><div><span class="sec-kicker">FEATURE MAP</span><h2>功能与工程落点</h2></div><span class="quiet-label">AI · SE · 核心功能</span></div><div class="requirement-row" v-for="r in engineering.requirements" :key="r.id"><span class="requirement-id">{{r.id}}</span><div><h3>{{r.name}}</h3><p>{{r.detail}}</p></div><span class="badge" :class="r.status==='已接入'?'approved':'pending'">{{r.status}}</span><button class="icon-button" @click="r.page==='accounts'?manageUsers():go(r.page)" :aria-label="`打开${r.name}`"><ArrowUpRight :size="18"/></button></div></section>
     <section class="criteria-section" v-reveal><div class="section-heading"><div><span class="sec-kicker">ACCEPTANCE</span><h2>验收标准与实际证据</h2></div><span class="quiet-label">保留现有SRS阈值</span></div><div class="data-table-wrap"><table><thead><tr><th>验收项</th><th>要求</th><th>当前证据</th><th>状态</th></tr></thead><tbody><tr v-for="c in engineering.criteria" :key="c.id"><td><strong>{{c.name}}</strong><small>{{c.id}}</small></td><td>{{c.target}}</td><td>{{typeof c.actual==='string'?c.actual:JSON.stringify(c.actual)}}</td><td><span class="badge pending">{{c.status}}</span></td></tr></tbody></table></div></section>

    </template>
    </div>
   </div>
  </main>
  <Transition name="toast"><div v-if="toast" class="toast" role="status"><CheckCircle2 :size="18"/>{{toast}}</div></Transition>
  <div v-if="showAuth" class="dialog-backdrop" @click.self="showAuth=false"><section ref="authDialog" class="auth-dialog" role="dialog" aria-modal="true" aria-labelledby="auth-title">
   <button class="icon-button dialog-close" @click="showAuth=false" aria-label="关闭登录"><X :size="20"/></button>
   <div class="auth-visual">
    <span class="auth-brand-mark"><Flame :size="26" :stroke-width="1.9"/></span>
    <h2>焰序工作台</h2>
    <p class="auth-tagline">从物理仿真到人工审核，合成火焰数据的完整证据链。</p>
    <ul class="auth-points">
     <li><CheckCircle2 :size="15"/>FDS 三维场 → Blender 渲染 → SD 受控增强</li>
     <li><CheckCircle2 :size="15"/>候选标注、人工审核与 Label Studio 同步</li>
     <li><CheckCircle2 :size="15"/>DVC 版本冻结与独立效果评价</li>
    </ul>
    <div class="auth-status"><span class="status-dot" :class="{online:settings?.worker.online}"></span><span>{{auth.require_login&&!auth.user?'登录后查看工作进程状态':settings?.worker.online?'工作进程已连接':'工作进程未连接'}}</span><span v-if="settings?.database" class="badge neutral">{{settings.database}}</span></div>
   </div>
   <div class="auth-side">
    <div v-if="!auth.setup_required" class="auth-tabs" role="tablist" aria-label="登录或注册"><button type="button" role="tab" :aria-selected="authTab==='login'" :class="{active:authTab==='login'}" @click="authTab='login'">登录</button><button v-if="auth.registration_open!==false" type="button" role="tab" :aria-selected="authTab==='register'" :class="{active:authTab==='register'}" @click="authTab='register'">注册</button></div>
    <template v-if="authTab==='login'&&!auth.setup_required">
     <h3 id="auth-title">登录工作区</h3>
     <p class="auth-desc">登录后可以提交任务、审核样本并发布研究快照。</p>
     <form class="stacked-form" @submit.prevent="authenticate">
      <label>用户名<input v-model="loginForm.username" required minlength="3" maxlength="32" autocomplete="username" autofocus placeholder="例如：researcher"></label>
      <label>密码<input v-model="loginForm.password" type="password" required minlength="10" maxlength="256" autocomplete="current-password" placeholder="至少10位"></label>
      <div v-if="authError" class="message error">{{authError}}</div>
      <button class="primary" :disabled="busy">{{busy?'正在验证…':'登录'}}</button>
     </form>
    </template>
    <template v-else>
     <h3 id="auth-title">{{auth.setup_required?'创建工作区':'注册工作区账户'}}</h3>
     <p class="auth-desc">{{auth.setup_required?'首次使用：自行设置管理员用户名和密码，账户自动拥有管理员权限。':'自助注册即时生效。五类角色都可提交任务与审核；管理员额外可创建项目与管理成员。'}}</p>
     <form class="stacked-form" @submit.prevent="register">
      <label>用户名<input v-model="registerForm.username" required minlength="3" maxlength="32" autocomplete="username" placeholder="3–32位字母、数字或下划线"></label>
      <label>密码<input v-model="registerForm.password" type="password" required minlength="10" maxlength="256" autocomplete="new-password" placeholder="至少10位"></label>
      <label v-if="!auth.setup_required">角色<select v-model="registerForm.role"><option v-for="(name,id) in roleNames" :key="id" :value="id">{{name}}</option></select></label>
      <div v-if="authError" class="message error">{{authError}}</div>
      <button class="primary" :disabled="busy">{{busy?'正在创建…':auth.setup_required?'创建并进入工作区':'注册并进入工作区'}}</button>
     </form>
    </template>
   </div>
  </section></div>
 </div>
</template>


