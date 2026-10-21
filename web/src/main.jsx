import React, { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { createRoot } from "react-dom/client";
import {
  Activity, ArrowDownToLine, ArrowLeft, ArrowUpRight, Boxes, Camera, Check, CheckCircle2, ChevronRight,
  CircleAlert, Copy, Database, ExternalLink, FileText, Flame, FolderOpen, Layers3, Library,
  ListChecks, LoaderCircle, LogOut, Pencil, Play, Plus, RefreshCw, RotateCcw, Save, Trash2, Upload, X,
} from "lucide-react";
import "./app.css";
import { AssetThumb, Panel, SceneWorkstation, assetRecord, boundsOfGas, defaultPlacements, liveLayout, pieceIssues, placeAsset, placeAssetAt, sameGas, samePose } from "./scene-editor.jsx";

const DEFAULT_CATALOG = {
  scenarios: ["房间火灾", "林区火灾", "工业区火灾"],
  scenes: ["室内", "室外"], weathers: ["晴", "雨", "雪"],
  sd_styles: { 室内: [], 室外: [] },
  viewpoints: ["俯视", "侧视", "第一人称"],
  augmentations: ["噪声", "模糊", "光照变化"],
  fuels: ["木材", "汽油", "塑料", "植被", "化学品"],
  assets: [], defaults: {}, shells: {},
  max_placements: 8, max_structures: 24, domain_cap_m: 4, shaft_half_m: 1,
  frame_count: { default: 16, max: 48 },
  gas_limits: {
    auto_cap_m: 4, min_edge_m: 0.5, resolution: [16, 128], noise: [0, 2, 3], max_bake_seconds: 1800,
    defaults: { warmup_s: 1, noise: 2, vorticity: 0.22, burning_rate: 0.6, flame_temp: 3, smoke: 1, samples: 20 },
    ranges: { warmup_s: [0.5, 10], vorticity: [0, 1], burning_rate: [0.1, 2], flame_temp: [1, 5], smoke: [0, 3], samples: [8, 256] },
  },
};
const INITIAL_GENERATION = {
  wind_speed: 1.5, temperature: 28, humidity: 45,
  weathers: ["晴"], viewpoints: ["侧视"],
  frame_count: 16, augmentations: [],
  min_flame_area: 0.002, seed: "", count: 1, wind_step: 0,
};
const GENERATION_KEY = "firephantom.generation.v2";
const PRESET_KEY = "firephantom.generation.presets.v2";
const DATASET_KEY = "firephantom.dataset.v1";
const PROPS_TAB_KEY = "firephantom.editor.tab.v1";
const NAV_ITEMS = [
  { id: "scenes", label: "场景库", icon: Boxes },
  { id: "library", label: "模型库", icon: Library },
  { id: "jobs", label: "作业", icon: Activity },
  { id: "datasets", label: "数据集与版本", icon: Database },
];
const SCENE_PRESETS = [
  { id: "房间火灾", cover: "/static/preset-covers/room.jpg", blurb: "沙发、木桌和盆栽围着起火点" },
  { id: "林区火灾", cover: "/static/preset-covers/forest.jpg", blurb: "松树和灌木围着地面的火" },
  { id: "工业区火灾", cover: "/static/preset-covers/industrial.jpg", blurb: "油桶和木箱围着油池" },
];
const LIBRARY_GROUPS = ["全部", "家具", "树", "灌木与草", "石头", "倒木", "容器", "我的上传", "其他"];

function todayStamp() {
  const now = new Date();
  const pad = (value) => String(value).padStart(2, "0");
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
}

function sizeText(size) {
  if (!Array.isArray(size) || size.length < 3 || size.some((value) => !Number.isFinite(Number(value)))) return "尺寸未知";
  return `${size.map((value) => Number(value).toFixed(2)).join(" × ")} m`;
}

function assetGroup(asset) {
  if (String(asset?.asset_id || "").startsWith("upload:")) return "我的上传";
  return asset?.group || "其他";
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { ...(options.body && typeof options.body === "string" ? { "Content-Type": "application/json" } : {}), ...options.headers },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : `请求失败 (${response.status})`);
  return payload;
}
const post = (path, body = {}) => api(path, { method: "POST", body: JSON.stringify(body) });
const put = (path, body) => api(path, { method: "PUT", body: JSON.stringify(body) });

function readStored(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}
function writeStored(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private window */ }
}

function useHashRoute() {
  const read = () => window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
  const [parts, setParts] = useState(read);
  useEffect(() => {
    const changed = () => setParts(read());
    window.addEventListener("hashchange", changed);
    return () => window.removeEventListener("hashchange", changed);
  }, []);
  return parts;
}
function go(...parts) {
  window.location.hash = `#/${parts.map(encodeURIComponent).join("/")}`;
}

function formatDate(value) {
  if (!value) return "-";
  return new Intl.DateTimeFormat("zh-CN", { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
}
function formatBytes(bytes = 0) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
function jobLabel(state) {
  return ({ queued: "排队中", running: "运行中", succeeded: "已完成", partial: "部分完成", failed: "失败", cancelled: "已取消" })[state] || state;
}
function statusLabel(status) {
  return ({ generated: "可送审", pending_review: "审核中", dropped: "已筛除", reviewed: "已审核", rejected: "已拒绝" })[status] || status || "";
}
function ordered(options, selected) {
  return (options || []).filter((item) => (selected || []).includes(item));
}
function memoryState(status) {
  return status?.memory?.state || "unknown";
}
function sampleFile(sampleId, path) {
  return `/api/v1/samples/${encodeURIComponent(sampleId)}/files/${path}`;
}
function versionFileUrl(versionId, path) {
  return `/api/v1/versions/${encodeURIComponent(versionId)}/files/${path.split("/").map(encodeURIComponent).join("/")}`;
}
function ratioText(map) {
  if (!map) return "-";
  return Object.entries(map).map(([name, value]) => `${name} ${(Number(value.ratio) * 100).toFixed(1)}%`).join(" · ");
}
function generationSize(catalog, form) {
  const weathers = ordered(catalog.weathers, form.weathers);
  const viewpoints = ordered(catalog.viewpoints, form.viewpoints);
  const combos = weathers.length * viewpoints.length;
  const perCombo = Number(form.frame_count) * (1 + (form.augmentations || []).length);
  return { weathers, viewpoints, combos, samples: combos * perCombo * Math.max(1, Number(form.count) || 1) };
}
function jobItem(catalog, form, sceneId) {
  const size = generationSize(catalog, form);
  return {
    scene_id: sceneId,
    fire_engine: "blender",
    wind_speed: Number(form.wind_speed),
    temperature: Number(form.temperature),
    humidity: Number(form.humidity),
    weathers: size.weathers,
    viewpoints: size.viewpoints,
    frame_count: Number(form.frame_count),
    augmentations: form.augmentations,
    min_flame_area: Number(form.min_flame_area),
    seed: String(form.seed).trim() === "" ? null : Number(form.seed),
    count: Number(form.count) || 1,
    wind_step: Number(form.wind_step) || 0,
  };
}
function cleanPlacements(list) {
  return (list || []).map((item) => {
    const row = {
      asset_id: item.asset_id, name: item.name,
      x: Number(item.x), y: Number(item.y), z: Number(item.z), yaw: Number(item.yaw) || 0,
      fuel_type: item.fuel_type, ignition: Boolean(item.ignition),
    };
    if (String(item.asset_id || "").startsWith("upload:")) row.size = item.size;
    return row;
  });
}
function gasBox(gas) {
  if (!gas?.location || !gas?.scale) return null;
  return { location: gas.location, scale: gas.scale, resolution: Number(gas.resolution) || 32 };
}
function sceneContent(scene) {
  if (!scene) return "";
  return JSON.stringify([
    scene.scenario,
    scene.origin_fuel,
    cleanPlacements(scene.placements),
    scene.origin || null,
    scene.cameras || {},
    gasBox(scene.gas_domain),
    scene.lights ?? null,
    scene.simulation || null,
    scene.structures ?? null,
    scene.domain || null,
  ]);
}

function App({ session, onLogout }) {
  const route = useHashRoute();
  const page = route[0] || "scenes";
  const [catalog, setCatalog] = useState(DEFAULT_CATALOG);
  const [uploads, setUploads] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [datasetId, setDatasetIdState] = useState(() => readStored(DATASET_KEY, ""));
  const [generation, setGenerationState] = useState(() => ({ ...INITIAL_GENERATION, ...readStored(GENERATION_KEY, {}) }));
  const [drafts, setDrafts] = useState({});
  const [systemStatus, setSystemStatus] = useState(null);
  const [pageError, setPageError] = useState("");
  const [toast, setToast] = useState("");
  const [mobileMenu, setMobileMenu] = useState(false);
  const setDatasetId = (value) => { setDatasetIdState(value); writeStored(DATASET_KEY, value); };
  const setGeneration = (updater) => setGenerationState((current) => {
    const next = typeof updater === "function" ? updater(current) : updater;
    writeStored(GENERATION_KEY, next);
    return next;
  });
  const refreshJobs = useCallback(async () => setJobs((await api("/api/v1/jobs")).jobs || []), []);
  const refreshDatasets = useCallback(async () => setDatasets((await api("/api/v1/datasets")).datasets || []), []);
  const refreshUploads = useCallback(async () => setUploads(((await api("/api/v1/assets/uploads")).uploads || []).map((item) => ({ ...item, fuel_type: item.fuel_type || "木材" }))), []);
  const fail = useCallback((error) => setPageError(error?.message || String(error)), []);
  const notify = useCallback((text) => setToast(text), []);

  useEffect(() => {
    api("/api/v1/catalog").then((data) => setCatalog({ ...DEFAULT_CATALOG, ...data })).catch(() => {});
    refreshUploads().catch(() => {});
    refreshDatasets().catch(fail);
    refreshJobs().catch(fail);
    api("/api/v1/system/status").then(setSystemStatus).catch(() => {});
  }, [fail, refreshDatasets, refreshJobs, refreshUploads]);
  const running = jobs.some((job) => ["queued", "running"].includes(job.state));
  useEffect(() => {
    const timer = window.setInterval(() => {
      refreshJobs().catch(() => {});
      api("/api/v1/system/status").then(setSystemStatus).catch(() => {});
    }, running ? 2500 : 8000);
    return () => window.clearInterval(timer);
  }, [refreshJobs, running]);
  useEffect(() => {
    if (!toast) return undefined;
    const timer = window.setTimeout(() => setToast(""), 4000);
    return () => window.clearTimeout(timer);
  }, [toast]);
  useEffect(() => { setMobileMenu(false); setPageError(""); }, [page]);
  useEffect(() => {
    if (datasetId && datasets.length && !datasets.some((item) => item.dataset_id === datasetId)) setDatasetId("");
  }, [datasets, datasetId]);

  const admit = async (sampleIds) => {
    if (!datasetId) { setPageError("先在顶栏选择当前数据集"); return false; }
    if (!sampleIds.length) return false;
    try {
      await post(`/api/v1/datasets/${encodeURIComponent(datasetId)}/samples`, { sample_ids: sampleIds });
      await refreshDatasets();
      notify(`已把 ${sampleIds.length} 个样本送去标注`);
      return true;
    } catch (error) { fail(error); return false; }
  };
  const logout = async () => {
    try { await api("/api/v1/auth/logout", { method: "POST" }); } catch (error) { fail(error); }
    onLogout();
  };
  const shared = { catalog, jobs, datasets, datasetId, refreshJobs, refreshDatasets, fail, notify, admit };
  const dataset = datasets.find((item) => item.dataset_id === datasetId);
  const crumb = page === "scenes" && route[1] ? "工作台" : NAV_ITEMS.find((item) => item.id === page)?.label || "场景库";
  const warning = ["critical", "warning"].includes(memoryState(systemStatus));
  const editing = page === "scenes" && Boolean(route[1]);
  const datasetPicker = <label className="dataset-pill" title="送审和发布都用这个数据集"><Database size={14} /><select value={datasetId} onChange={(event) => setDatasetId(event.target.value)} aria-label="当前数据集"><option value="">未选数据集</option>{datasets.map((item) => <option key={item.dataset_id} value={item.dataset_id}>{item.name}</option>)}</select>{dataset && <small>{dataset.samples} 样本</small>}</label>;
  const account = <><span className="account-pill">{session.email}</span><button className="icon-button" onClick={logout} title="退出" aria-label="退出"><LogOut size={16} /></button></>;

  return <div className={editing ? "app-shell editor-mode" : "app-shell"}>
    <aside className={`sidebar ${mobileMenu ? "sidebar-open" : ""}`}>
      <a className="brand" href="#/scenes" aria-label="FirePhantom 首页"><span className="brand-mark"><Flame size={18} strokeWidth={2.4} /></span><span>FirePhantom<small>DATASET LAB</small></span></a>
      <div className="workspace-label">工作区</div>
      <nav className="side-nav" aria-label="主导航">{NAV_ITEMS.map(({ id, label, icon: Icon }) => <a className={page === id ? "nav-item active" : "nav-item"} key={id} href={`#/${id}`}><Icon size={17} /><span>{label}</span>{id === "jobs" && running && <i className="nav-dot" />}</a>)}
        <a className="nav-item" href="/auth/label-studio" target="_blank" rel="noreferrer"><ExternalLink size={17} /><span>Label Studio</span></a>
      </nav>
      <div className="sidebar-spacer" />
      {session.is_admin && <AccountForm fail={fail} notify={notify} />}
      <div className="system-status"><span className={`status-led ${warning ? "status-led-warning" : ""}`} /><div><strong>{memoryState(systemStatus) === "critical" ? "内存不足" : warning ? "内存紧张" : "API 服务在线"}</strong><small>{systemStatus?.memory?.available_gib == null ? "资源状态未知" : `${systemStatus.memory.available_gib.toFixed(1)} GiB 可用`}</small></div></div>
      <a className="docs-link" href="/docs" target="_blank" rel="noreferrer"><FileText size={15} /> API 文档 <ArrowUpRight size={13} /></a>
    </aside>
    {mobileMenu && <button className="scrim" onClick={() => setMobileMenu(false)} aria-label="关闭导航" />}
    <main className="main-area">
      <header className="topbar">
        <button className="icon-button mobile-menu-button" onClick={() => setMobileMenu((open) => !open)} aria-label="打开导航"><Layers3 size={19} /></button>
        <div className="crumb"><span>FirePhantom</span><ChevronRight size={14} /><strong>{crumb}</strong></div>
        <div className="top-actions">
          {datasetPicker}
          {account}
        </div>
      </header>
      <div className="content-area">
        {pageError && <div className={editing ? "error-banner floating" : "error-banner"}><CircleAlert size={16} /><span>{pageError}</span><button className="icon-button" onClick={() => setPageError("")} aria-label="关闭"><X size={15} /></button></div>}
        {page === "scenes" && !route[1] && <ScenesPage {...shared} />}
        {page === "scenes" && route[1] && <WorkbenchPage key={route[1]} sceneId={route[1]} {...shared} uploads={uploads} setUploads={setUploads} generation={generation} setGeneration={setGeneration} drafts={drafts} setDrafts={setDrafts} headerEnd={<>{datasetPicker}{account}</>} />}
        {page === "library" && <LibraryPage catalog={catalog} uploads={uploads} setUploads={setUploads} fail={fail} notify={notify} />}
        {page === "jobs" && <JobsPage {...shared} jobId={route[1] || null} />}
        {page === "datasets" && <DatasetsPage {...shared} setDatasetId={setDatasetId} />}
      </div>
    </main>
    {toast && <div className="toast"><CheckCircle2 size={17} />{toast}</div>}
  </div>;
}

function AccountForm({ fail, notify }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const submit = async (event) => {
    event.preventDefault();
    try {
      const created = await post("/api/v1/auth/users", { email, password });
      setEmail(""); setPassword(""); notify(`已添加 ${created.email}`);
    } catch (error) { fail(error); }
  };
  return <form className="account-form" onSubmit={submit}><span>添加账号</span><input type="email" required placeholder="邮箱" value={email} onChange={(event) => setEmail(event.target.value)} /><input type="password" required minLength={8} placeholder="密码" value={password} onChange={(event) => setPassword(event.target.value)} /><button type="submit">添加</button></form>;
}

function ScenesPage({ catalog, jobs, fail, notify }) {
  const [scenes, setScenes] = useState(null);
  const [name, setName] = useState("");
  const [scenario, setScenario] = useState("");
  const [renaming, setRenaming] = useState(null);
  const refresh = useCallback(async () => setScenes((await api("/api/v1/scenes")).scenes || []), []);
  useEffect(() => { refresh().catch(fail); }, [fail, refresh]);
  const create = async (event) => {
    event.preventDefault();
    try {
      const scene = await post("/api/v1/scenes", {
        name: name.trim() || `${scenario} ${todayStamp()}`,
        scenario,
        origin_fuel: scenario === "林区火灾" ? "植被" : scenario === "工业区火灾" ? "汽油" : "木材",
        placements: cleanPlacements(defaultPlacements(catalog, scenario)),
      });
      go("scenes", scene.scene_id);
    } catch (error) { fail(error); }
  };
  const duplicate = async (scene) => {
    try { await post(`/api/v1/scenes/${scene.scene_id}/duplicate`, {}); await refresh(); notify(`已复制「${scene.name}」`); } catch (error) { fail(error); }
  };
  const remove = async (scene) => {
    if (!window.confirm(`从场景库删除「${scene.name}」？已经生成的样本和作业记录不受影响。`)) return;
    try { await api(`/api/v1/scenes/${scene.scene_id}`, { method: "DELETE" }); await refresh(); } catch (error) { fail(error); }
  };
  const rename = async (scene, value) => {
    setRenaming(null);
    if (!value.trim() || value.trim() === scene.name) return;
    try { await put(`/api/v1/scenes/${scene.scene_id}`, { name: value.trim() }); await refresh(); } catch (error) { fail(error); }
  };
  const runs = (sceneId) => jobs.filter((job) => (job.scene_ids || []).includes(sceneId));
  const presets = SCENE_PRESETS.filter((item) => catalog.scenarios.includes(item.id));
  const chosen = presets.find((item) => item.id === scenario);
  const pickPreset = (id) => {
    setScenario(id);
    setName(`${id} ${todayStamp()}`);
  };
  return <>
    <div className="page-heading"><div><div className="eyebrow">SCENE LIBRARY</div><h1>场景库</h1><p>一个场景是一套燃料摆放。新建时先选预设，再打开同一页调参数、生成、看结果。场景改动按版本保存，作业记住用的是哪一版。</p></div></div>
    <section className="surface wizard" aria-label="新建场景">
      {chosen ? (
        <form className="wizard-name" onSubmit={create}>
          <button type="button" className="secondary-button" onClick={() => setScenario("")}>返回</button>
          <label className="field"><span className="field-label">{chosen.id}</span><input value={name} maxLength={120} aria-label="新场景名称" onChange={(event) => setName(event.target.value)} /></label>
          <button className="primary-button" type="submit"><Plus size={16} /> 创建并打开</button>
        </form>
      ) : (
        <div className="wizard-cards">
          {presets.map((item) => (
            <button type="button" className="wizard-card" key={item.id} onClick={() => pickPreset(item.id)}>
              <img src={item.cover} alt="" />
              <strong>{item.id}</strong>
              <span>{item.blurb}</span>
            </button>
          ))}
        </div>
      )}
    </section>
    {scenes === null ? <div className="empty-table"><LoaderCircle className="spin" size={20} /></div> : !scenes.length ? <EmptyState icon={Boxes} title="还没有场景" message="新建一个，默认带上这个预设的燃料摆放。" /> : <div className="scene-grid">{scenes.map((scene) => {
      const history = runs(scene.scene_id);
      return <article className="scene-card" key={scene.scene_id}>
        <a className="scene-card-thumb" href={`#/scenes/${scene.scene_id}`}>{scene.thumbnail_url ? <img src={scene.thumbnail_url} alt="" /> : <span><Boxes size={22} /></span>}</a>
        <div className="scene-card-body">
          {renaming === scene.scene_id
            ? <input className="scene-rename" autoFocus defaultValue={scene.name} maxLength={120} onBlur={(event) => rename(scene, event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") event.currentTarget.blur(); if (event.key === "Escape") setRenaming(null); }} />
            : <a className="scene-card-title" href={`#/scenes/${scene.scene_id}`}>{scene.name}</a>}
          <small>预设 {scene.scenario} · {scene.placements.length} 件 · 第 {scene.revision} 版</small>
          <small>{history.length ? `${history.length} 次生成 · 最近 ${formatDate(history[0].created_at)}` : `更新于 ${formatDate(scene.updated_at)}`}</small>
        </div>
        <div className="scene-card-actions">
          <button className="icon-button" title="重命名" aria-label="重命名" onClick={() => setRenaming(scene.scene_id)}><Pencil size={14} /></button>
          <button className="icon-button" title="复制" aria-label="复制" onClick={() => duplicate(scene)}><Copy size={14} /></button>
          <button className="icon-button" title="删除" aria-label="删除" onClick={() => remove(scene)}><Trash2 size={14} /></button>
        </div>
      </article>;
    })}</div>}
  </>;
}

function LibraryPage({ catalog, uploads, setUploads, fail, notify }) {
  const [query, setQuery] = useState("");
  const [group, setGroup] = useState("全部");
  const [busy, setBusy] = useState(false);
  const [over, setOver] = useState(false);
  const [report, setReport] = useState(null);
  const fileRef = useRef(null);
  const builtins = catalog.assets || [];
  const mine = uploads || [];
  const needle = query.trim().toLowerCase();
  const match = (asset) => {
    const bucket = assetGroup(asset);
    if (group !== "全部" && bucket !== group) return false;
    if (!needle) return true;
    return `${asset.label || ""} ${bucket} ${asset.fuel_type || ""} ${asset.note || ""}`.toLowerCase().includes(needle);
  };
  const shownBuiltins = builtins.filter(match);
  const shownMine = mine.filter(match);
  const groups = LIBRARY_GROUPS.filter((name) => name === "全部" || name === "我的上传" || builtins.some((asset) => assetGroup(asset) === name) || mine.some((asset) => assetGroup(asset) === name));
  const send = async (fileList) => {
    const files = Array.from(fileList || []).filter((file) => {
      const name = file.name.toLowerCase();
      return name.endsWith(".glb") || name.endsWith(".zip");
    });
    if (!files.length) {
      setReport({ accepted: [], rejected: [{ name: "选择的文件", reason: "只接受 .glb，或装着 .glb 的 .zip" }] });
      return;
    }
    setBusy(true);
    setReport(null);
    try {
      const body = new FormData();
      files.forEach((file) => body.append("files", file));
      const response = await fetch("/api/v1/assets/uploads/batch", { method: "POST", body });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "上传失败");
      const accepted = (payload.accepted || []).map((item) => ({ ...item, fuel_type: item.fuel_type || "木材" }));
      setUploads((current) => {
        const ids = new Set(accepted.map((item) => item.asset_id));
        return [...accepted, ...current.filter((item) => !ids.has(item.asset_id))];
      });
      setReport({ accepted, rejected: payload.rejected || [] });
      const failed = (payload.rejected || []).length;
      notify(failed ? `放入 ${accepted.length} 个，${failed} 个没放进去` : `放入 ${accepted.length} 个模型`);
    } catch (error) {
      fail(error);
    } finally {
      setBusy(false);
    }
  };
  const remove = async (asset) => {
    const token = String(asset.asset_id || "").split(":")[1];
    if (!token) return;
    if (!window.confirm(`从模型库删除「${asset.label || "上传"}」？已经摆进场景的会改画成燃料盒。`)) return;
    try {
      await api(`/api/v1/assets/uploads/${encodeURIComponent(token)}`, { method: "DELETE" });
      setUploads((current) => current.filter((item) => item.asset_id !== asset.asset_id));
      notify(`已删除「${asset.label || "上传"}」`);
    } catch (error) {
      fail(error);
    }
  };
  const card = (asset, own) => (
    <article className="library-card" key={asset.asset_id}>
      <AssetThumb asset={asset} className="library-thumb" />
      <div>
        <strong title={asset.label}>{asset.label}</strong>
        <small>{assetGroup(asset)} · {asset.fuel_type || "木材"}</small>
        <small>{sizeText(asset.size)}</small>
        {asset.note ? <small className="library-note">{asset.note}</small> : null}
      </div>
      {own ? <button type="button" className="icon-button" title="删除" aria-label={`删除${asset.label || "上传"}`} onClick={() => remove(asset)}><Trash2 size={14} /></button> : <span className="library-lock">内置</span>}
    </article>
  );
  return <>
    <div className="page-heading"><div><div className="eyebrow">MODEL LIBRARY</div><h1>模型库</h1><p>内置模型是 CC0，可以商用。求解器烧的是包围盒，不是网格本身。一次可以选很多 .glb，或一个装着 .glb 的 zip。每个文件不超过 8 MB、5 万面。</p></div></div>
    <section
      className={over ? "surface library-drop over" : "surface library-drop"}
      onDragOver={(event) => { event.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(event) => { event.preventDefault(); setOver(false); send(event.dataTransfer.files); }}
    >
      <div>
        <strong>批量放入模型</strong>
        <p>把 .glb 或 zip 拖到这里，也可以一次选多个文件。放不进去的会写明原因，其余的仍然留下。</p>
      </div>
      <button type="button" className="primary-button" disabled={busy} onClick={() => fileRef.current?.click()}><Upload size={16} />{busy ? "上传中" : "选择文件"}</button>
      <input ref={fileRef} type="file" accept=".glb,.zip,model/gltf-binary,application/zip" multiple hidden onChange={(event) => { send(event.target.files); event.target.value = ""; }} />
    </section>
    {report && (
      <div className="library-report" role="status">
        <span>本次放入 {report.accepted.length} 个{report.rejected.length ? `，${report.rejected.length} 个没放进去` : "。"}</span>
        {report.rejected.map((item) => <small key={`${item.name}:${item.reason}`}>{item.name}：{item.reason}</small>)}
      </div>
    )}
    <div className="library-tools">
      <input value={query} placeholder="搜索名称、分组或燃料" aria-label="搜索模型" onChange={(event) => setQuery(event.target.value)} />
      <div className="library-groups" role="group" aria-label="分组">
        {groups.map((name) => <button type="button" key={name} className={group === name ? "on" : ""} onClick={() => setGroup(name)}>{name}</button>)}
      </div>
    </div>
    <h2 className="library-heading">内置 <small>{shownBuiltins.length}</small></h2>
    {shownBuiltins.length ? <div className="library-grid">{shownBuiltins.map((asset) => card(asset, false))}</div> : <p className="library-empty">没有匹配的内置模型。</p>}
    <h2 className="library-heading">我的上传 <small>{shownMine.length}</small></h2>
    {shownMine.length ? <div className="library-grid">{shownMine.map((asset) => card(asset, true))}</div> : <p className="library-empty">{mine.length ? "没有匹配的上传。" : "还没有上传。拖几个 .glb 进来就会出现在这里，也可以拖进场景。"}</p>}
  </>;
}

function WorkbenchPage({ sceneId, catalog, jobs, uploads, setUploads, generation, setGeneration, drafts, setDrafts, refreshJobs, fail, notify, admit, datasetId, headerEnd }) {
  const [server, setServer] = useState(null);
  const [piece, setPiece] = useState({ kind: "piece", index: 0 });
  const [preview, setPreview] = useState(null);
  const [tool, setTool] = useState("translate");
  const [saving, setSaving] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [openJob, setOpenJob] = useState(null);
  const [propsTab, setPropsTabState] = useState(() => readStored(PROPS_TAB_KEY, "object"));
  const setPropsTab = (value) => { setPropsTabState(value); writeStored(PROPS_TAB_KEY, value); };
  const canvas = useRef(null);
  const draft = drafts[sceneId] || server;
  const dirty = Boolean(server && draft && (sceneContent(draft) !== sceneContent(server) || draft.name !== server.name));
  useEffect(() => {
    api(`/api/v1/scenes/${encodeURIComponent(sceneId)}`).then(setServer).catch((error) => { fail(error); go("scenes"); });
  }, [fail, sceneId]);
  useEffect(() => {
    if (!dirty) return undefined;
    const warn = (event) => { event.preventDefault(); event.returnValue = ""; };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);
  const sceneJobs = jobs.filter((job) => (job.scene_ids || []).includes(sceneId));
  useEffect(() => {
    if (!openJob && sceneJobs[0]) setOpenJob(sceneJobs[0].job_id);
  }, [openJob, sceneJobs]);
  const previewBody = draft ? {
    scenario: draft.scenario,
    origin_fuel: draft.origin_fuel,
    placements: cleanPlacements(draft.placements),
    origin: draft.origin || null,
    cameras: draft.cameras || {},
    gas_domain: gasBox(draft.gas_domain),
    simulation: draft.simulation || null,
    structures: draft.structures ?? null,
    domain: draft.domain || null,
    viewpoints: ordered(catalog.viewpoints, generation.viewpoints),
    frame_count: Number(generation.frame_count) || 16,
    // Each weather is its own bake.
    look_count: Math.max(1, generationSize(catalog, generation).weathers.length),
  } : null;
  const layoutKey = previewBody ? JSON.stringify(previewBody) : "";
  useEffect(() => {
    if (!layoutKey) return undefined;
    const controller = new AbortController();
    const sent = JSON.parse(layoutKey);
    const timer = window.setTimeout(() => {
      api("/api/v1/layout/preview", { method: "POST", body: layoutKey, signal: controller.signal }).then((data) => {
        if (controller.signal.aborted) return;
        setPreview(data);
        const snapped = gasBox(data.gas_domain);
        if (sent.gas_domain && data.gas_overridden && snapped && !sameGas(sent.gas_domain, snapped)) {
          notify("气体域贴着地面或墙，已自动让开约 5 cm");
        }
        const buried = (data.rejected_cameras || []).filter((view) => sent.cameras?.[view]);
        if (buried.length) notify(`${buried.join("、")}相机在墙或天花板里面，已改回自动机位`);
        setDrafts((current) => {
          const row = current[sceneId] || server;
          if (!row) return current;
          const rowKey = JSON.stringify({
            scenario: row.scenario,
            origin_fuel: row.origin_fuel,
            placements: cleanPlacements(row.placements),
            origin: row.origin || null,
            cameras: row.cameras || {},
            gas_domain: gasBox(row.gas_domain),
            simulation: row.simulation || null,
            structures: row.structures ?? null,
            domain: row.domain || null,
            viewpoints: sent.viewpoints,
            frame_count: sent.frame_count,
            look_count: sent.look_count,
          });
          if (rowKey !== layoutKey) return current;
          const cameras = { ...(row.cameras || {}) };
          let camerasChanged = false;
          Object.keys(cameras).forEach((view) => {
            const auto = data.automatic_cameras?.[view];
            if ((auto && samePose(cameras[view], auto)) || buried.includes(view)) {
              delete cameras[view];
              camerasChanged = true;
            }
          });
          let gas = gasBox(row.gas_domain);
          const hadGas = gas != null;
          if (gas && data.gas_overridden === false) gas = null;
          else if (gas && data.gas_domain) {
            const snapped = gasBox(data.gas_domain);
            if (snapped && !sameGas(gas, snapped)) gas = snapped;
          }
          const gasChanged = (gas == null) !== !hadGas || Boolean(gas && hadGas && !sameGas(gas, gasBox(row.gas_domain)));
          if (!camerasChanged && !gasChanged) return current;
          const base = current[sceneId] || server;
          if (!base) return current;
          return {
            ...current,
            [sceneId]: {
              ...base,
              cameras: camerasChanged ? cameras : (base.cameras || {}),
              gas_domain: gasChanged ? gas : (base.gas_domain || null),
            },
          };
        });
      }).catch((error) => {
        if (error?.name === "AbortError") return;
      });
    }, 200);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [layoutKey, sceneId, server, setDrafts, notify]);
  if (!draft) return <div className="empty-table"><LoaderCircle className="spin" size={20} /><span>正在打开场景</span></div>;

  const edit = (patch) => setDrafts((current) => ({ ...current, [sceneId]: { ...(current[sceneId] || server), ...patch } }));
  const pieces = draft.placements || [];
  const shell = catalog.shells?.[draft.scenario];
  const layoutShell = liveLayout(shell, { structures: draft.structures ?? null, domain: draft.domain || null, origin: draft.origin || null });
  const shelf = [...(catalog.assets || []), ...uploads];
  const limit = catalog.max_placements || 8;
  const writePieces = (list) => edit({ placements: list });
  const addPiece = (asset, point) => {
    const payload = { ...asset, fuel_type: asset.fuel_type || "木材" };
    const result = point ? placeAssetAt(layoutShell, pieces, payload, limit, point.x, point.y) : placeAsset(layoutShell, pieces, payload, limit);
    if (result.error) return result.error;
    writePieces([...pieces, result.item]);
    setPiece({ kind: "piece", index: pieces.length });
    return "";
  };
  const duplicatePiece = (index) => {
    const source = pieces[index];
    if (!source) return "";
    const asset = assetRecord(shelf, source);
    const result = placeAsset(layoutShell, pieces, { ...asset, size: source.size, fuel_type: source.fuel_type || "木材" }, limit);
    if (result.error) return result.error;
    writePieces([...pieces, { ...result.item, yaw: source.yaw || 0, ignition: Boolean(source.ignition), fuel_type: source.fuel_type || "木材" }]);
    setPiece({ kind: "piece", index: pieces.length });
    return "";
  };
  const uploadPiece = async (file) => {
    const body = new FormData();
    body.append("file", file);
    const response = await fetch("/api/v1/assets/uploads", { method: "POST", body });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "上传失败");
    const asset = { ...payload, fuel_type: "木材" };
    setUploads((current) => [asset, ...current.filter((entry) => entry.asset_id !== asset.asset_id)]);
    return asset;
  };
  const issueShell = layoutShell || shell;
  const gasLimits = { ...DEFAULT_CATALOG.gas_limits, ...(catalog.gas_limits || {}) };
  const overBudget = Boolean(preview?.bake_estimate?.over);
  const rejected = pieces.filter((entry) => pieceIssues(entry, pieces, issueShell, gasLimits.auto_cap_m, boundsOfGas(draft.gas_domain || preview?.gas_domain)).some((code) => code === "shaft" || code === "outside")).length;
  const capture = () => new Promise((resolve) => {
    if (!canvas.current) { resolve(null); return; }
    canvas.current.toBlob((blob) => resolve(blob), "image/png");
  });
  const save = async () => {
    if (saving) return server;
    setSaving(true);
    try {
      let saved = server;
      if (dirty) {
        saved = await put(`/api/v1/scenes/${sceneId}`, {
          name: draft.name, scenario: draft.scenario, origin_fuel: draft.origin_fuel,
          placements: cleanPlacements(draft.placements),
          origin: draft.origin || null,
          cameras: draft.cameras || {},
          gas_domain: gasBox(draft.gas_domain),
          lights: draft.lights ?? null,
          simulation: draft.simulation || null,
          structures: draft.structures ?? null,
          domain: draft.domain || null,
          base_revision: server.revision,
        });
      }
      const blob = await capture();
      if (blob) saved = await api(`/api/v1/scenes/${sceneId}/thumbnail`, { method: "PUT", body: blob, headers: { "Content-Type": "image/png" } });
      setServer(saved);
      setDrafts((current) => { const next = { ...current }; delete next[sceneId]; return next; });
      return saved;
    } catch (error) { fail(error); return null; } finally { setSaving(false); }
  };
  const submit = async () => {
    setSubmitting(true);
    try {
      const saved = dirty ? await save() : server;
      if (!saved) return;
      const job = await post("/api/v1/jobs", { items: [jobItem(catalog, generation, sceneId)] });
      await refreshJobs();
      setOpenJob(job.job_id);
      setPropsTab("jobs");
      notify(`第 ${saved.revision} 版场景已开始生成`);
    } catch (error) { fail(error); } finally { setSubmitting(false); }
  };
  const header = <div className="workbench-title">
    <a className="be-icon" href="#/scenes" title="返回场景库" aria-label="返回场景库"><ArrowLeft size={15} /></a>
    <input className="scene-name-input" value={draft.name} maxLength={120} aria-label="场景名称" onChange={(event) => edit({ name: event.target.value })} />
    <span className={dirty ? "state-chip warning" : "state-chip success"}>{dirty ? "未保存" : `第 ${server.revision} 版`}</span>
    <button type="button" className="be-btn" title="保存 (⌘S)" disabled={saving} onClick={save}>{saving ? <LoaderCircle size={13} className="spin" /> : <Save size={13} />}保存</button>
  </div>;
  const tabs = [
    {
      id: "render",
      label: "渲染",
      icon: Camera,
      content: <GenerationSettings
        catalog={catalog}
        environment={preview?.environment || null}
        form={generation}
        setForm={setGeneration}
        submitting={submitting}
        blocked={rejected ? `${rejected} 件摆放会被拒绝，先挪开` : overBudget ? "气体模拟超出算力上限，先在气体域面板里调小" : ""}
        onSubmit={submit}
        notify={notify}
      />,
    },
    {
      id: "jobs",
      label: "作业",
      icon: ListChecks,
      badge: sceneJobs.some((job) => ["queued", "running"].includes(job.state)),
      content: <SceneRuns
        sceneJobs={sceneJobs}
        openJob={openJob}
        setOpenJob={setOpenJob}
        admit={admit}
        datasetId={datasetId}
        fail={fail}
        notify={notify}
        refreshJobs={refreshJobs}
      />,
    },
  ];
  return <SceneWorkstation
    scenario={draft.scenario}
    scenarios={catalog.scenarios}
    shell={shell}
    items={pieces}
    assets={shelf}
    fuels={catalog.fuels}
    selected={piece}
    tool={tool}
    domainCap={gasLimits.auto_cap_m}
    gasLimits={gasLimits}
    simulation={draft.simulation || null}
    onSimulation={(value) => edit({ simulation: value })}
    maxPlacements={limit}
    originFuel={draft.origin_fuel}
    originOverride={draft.origin || null}
    cameraOverrides={draft.cameras || {}}
    gasOverride={draft.gas_domain || null}
    lights={draft.lights ?? null}
    automaticLights={preview?.automatic_lights || []}
    maxLights={catalog.max_lights || 8}
    structures={draft.structures ?? null}
    domainOverride={draft.domain || null}
    maxStructures={catalog.max_structures || 24}
    preview={preview}
    viewpoints={ordered(catalog.viewpoints, generation.viewpoints)}
    header={header}
    headerEnd={headerEnd}
    tabs={tabs}
    activeTab={propsTab}
    onTab={setPropsTab}
    onApplyPreset={(value) => {
      edit({
        scenario: value,
        structures: null,
        domain: null,
        placements: cleanPlacements(defaultPlacements(catalog, value)),
        origin: null,
        cameras: {},
        gas_domain: null,
        lights: null,
      });
      setPiece({ kind: "piece", index: 0 });
      setPreview(null);
    }}
    onStructures={(value) => edit({ structures: value })}
    onDomain={(value) => edit({ domain: value })}
    onOriginFuel={(value) => edit({ origin_fuel: value })}
    onOrigin={(value) => edit({ origin: value })}
    onCameras={(value) => edit({ cameras: value })}
    onGas={(value) => edit({ gas_domain: value })}
    onLights={(value) => edit({ lights: value })}
    onTool={setTool}
    onSelect={setPiece}
    onCommit={(index, patch) => writePieces(pieces.map((item, i) => (i === index ? { ...item, ...patch } : item)))}
    onAdd={addPiece}
    onRemove={(index) => {
      writePieces(pieces.filter((_, i) => i !== index));
      setPiece((current) => {
        if (current?.kind !== "piece") return current;
        if (current.index === index) {
          const next = Math.min(index, pieces.length - 2);
          return next >= 0 ? { kind: "piece", index: next } : null;
        }
        if (current.index > index) return { kind: "piece", index: current.index - 1 };
        return current;
      });
    }}
    onDuplicate={duplicatePiece}
    onUpload={uploadPiece}
    onRestore={() => { writePieces(cleanPlacements(defaultPlacements(catalog, draft.scenario))); setPiece({ kind: "piece", index: 0 }); }}
    onCanvas={(element) => { canvas.current = element; }}
    onSave={save}
  />;
}

function readPresets() {
  const stored = readStored(PRESET_KEY, []);
  return Array.isArray(stored) ? stored.filter((item) => item && item.id && item.name && item.config) : [];
}

function GenerationSettings({ catalog, environment, form, setForm, submitting, blocked, onSubmit, notify }) {
  const [presets, setPresets] = useState(readPresets);
  const [presetName, setPresetName] = useState("");
  const set = (key, value) => setForm((current) => ({ ...current, [key]: value }));
  const toggle = (key, name) => set(key, form[key].includes(name) ? form[key].filter((item) => item !== name) : [...form[key], name]);
  const size = generationSize(catalog, form);
  const maxFrames = catalog.frame_count?.max || 48;
  const frames = Number(form.frame_count);
  const ready = size.combos > 0 && Number.isInteger(frames) && frames >= 1 && frames <= maxFrames && size.samples <= 20000 && !blocked;
  const skyHint = environment === "室内"
    ? "这一烘是室内天空：顶盖盖住了雨雪落下的范围。雨雪落在顶上，地面保持干燥。火焰仍跟着天气，雨更凉、烟更多，雪稍凉。"
    : environment === "室外"
      ? "这一烘是室外天空。雨雪进到场地里，地面会湿或积雪。火焰跟着天气，雨更凉、烟更多，雪稍凉。"
      : "天空跟着顶盖。顶盖盖住雨雪落下的范围时是室内光，否则是室外光。晴、雨、雪都可以选。";
  const savePreset = () => {
    const name = presetName.trim();
    if (!name) return;
    const next = [...presets.filter((item) => item.name !== name), { id: `p-${Date.now()}`, name, config: form }];
    setPresets(next); writeStored(PRESET_KEY, next); setPresetName(""); notify(`已保存生成预设「${name}」`);
  };
  const presetPicker = presets.length > 0 && <select className="preset-picker" value="" aria-label="生成预设" onChange={(event) => { const preset = presets.find((item) => item.id === event.target.value); if (preset) setForm({ ...INITIAL_GENERATION, ...preset.config }); }}><option value="">生成预设</option>{presets.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select>;
  return <>
    <Panel title="仿真" actions={presetPicker}>
      <Field label="风速" unit="m/s"><input type="number" min="0" max="100" step="0.1" value={form.wind_speed} onChange={(event) => set("wind_speed", event.target.value)} /></Field>
      <Field label="温度" unit="°C"><input type="number" min="-80" max="100" step="0.1" value={form.temperature} onChange={(event) => set("temperature", event.target.value)} /></Field>
      <Field label="湿度" unit="%"><input type="number" min="0" max="100" step="1" value={form.humidity} onChange={(event) => set("humidity", event.target.value)} /></Field>
    </Panel>
    <Panel title="输出" hint={skyHint}>
      <ChoiceGroup label="天气" options={catalog.weathers} selected={size.weathers} onToggle={(name) => toggle("weathers", name)} />
      <ChoiceGroup label="视角" options={catalog.viewpoints} selected={form.viewpoints} onToggle={(name) => toggle("viewpoints", name)} />
      <Field label="每视角帧数" unit={`1-${maxFrames}`}><input type="number" min="1" max={maxFrames} step="1" value={form.frame_count} onChange={(event) => set("frame_count", event.target.value)} /></Field>
      <ChoiceGroup label="增强（另存派生样本）" options={catalog.augmentations} selected={form.augmentations} onToggle={(name) => toggle("augmentations", name)} />
    </Panel>
    <Panel title="高级" defaultOpen={false}>
      <Field label="种子"><input type="number" min="0" step="1" value={form.seed} placeholder="按参数推导" onChange={(event) => set("seed", event.target.value)} /></Field>
      <Field label="最小火焰占比"><input type="number" min="0" max="0.5" step="0.001" value={form.min_flame_area} onChange={(event) => set("min_flame_area", event.target.value)} /></Field>
      <Field label="重复烘焙"><input type="number" min="1" max="100" step="1" value={form.count} onChange={(event) => set("count", event.target.value)} /></Field>
      <Field label="每次加风" unit="m/s"><input type="number" min="0" max="100" step="0.1" value={form.wind_step} onChange={(event) => set("wind_step", event.target.value)} /></Field>
      <div className="preset-save"><input value={presetName} maxLength={60} placeholder="把这组参数存成生成预设" onChange={(event) => setPresetName(event.target.value)} /><button type="button" className="secondary-button" onClick={savePreset}>保存</button></div>
    </Panel>
    <div className="be-submit">
      {blocked && <p className="stage-note">{blocked}</p>}
      {size.samples > 20000 && <p className="stage-note">一次最多 20000 个样本</p>}
      <button className="primary-button" type="button" disabled={!ready || submitting} onClick={onSubmit}>{submitting ? <LoaderCircle size={15} className="spin" /> : <Play size={14} fill="currentColor" />}生成 {size.combos} 组 · {size.samples} 个样本</button>
      <p className="field-hint">未保存的摆放会先保存成新版本再提交。</p>
    </div>
  </>;
}

function SceneRuns({ sceneJobs, openJob, setOpenJob, admit, datasetId, fail, notify, refreshJobs }) {
  const job = sceneJobs.find((item) => item.job_id === openJob);
  return <Panel title="这个场景的生成" actions={job && <a className="text-button" href={`#/jobs/${job.job_id}`}>在作业页打开</a>}>
    {!sceneJobs.length ? <p className="field-hint">还没有生成过。在「渲染」标签里提交。</p> : <div className="run-tabs">{sceneJobs.slice(0, 6).map((item) => <button type="button" key={item.job_id} className={item.job_id === openJob ? "run-tab selected" : "run-tab"} onClick={() => setOpenJob(item.job_id)}><StateDot state={item.state} />{formatDate(item.created_at)}<small>{item.completed}/{item.total}</small></button>)}</div>}
    {openJob && <JobDetail key={openJob} jobId={openJob} compact admit={admit} datasetId={datasetId} fail={fail} notify={notify} refreshJobs={refreshJobs} />}
  </Panel>;
}

function StateDot({ state }) {
  if (["queued", "running"].includes(state)) return <LoaderCircle size={12} className="spin" />;
  return <span className={`state-dot ${state}`} />;
}

function useJob(jobId) {
  const [job, setJob] = useState(null);
  const [stages, setStages] = useState(null);
  const refresh = useCallback(async () => {
    const [detail, view] = await Promise.all([
      api(`/api/v1/jobs/${encodeURIComponent(jobId)}?result_limit=1`),
      api(`/api/v1/jobs/${encodeURIComponent(jobId)}/stages`),
    ]);
    setJob(detail); setStages(view);
    return detail;
  }, [jobId]);
  useEffect(() => {
    let timer = null;
    let alive = true;
    const tick = async () => {
      try {
        const detail = await refresh();
        if (alive && ["queued", "running"].includes(detail.state)) timer = window.setTimeout(tick, 2000);
      } catch { if (alive) timer = window.setTimeout(tick, 5000); }
    };
    tick();
    return () => { alive = false; window.clearTimeout(timer); };
  }, [refresh]);
  return { job, stages, refresh };
}

function admissible(sample, datasetId) {
  return sample.status === "generated" && (!sample.dataset_id || sample.dataset_id === datasetId);
}

function JobDetail({ jobId, compact = false, admit, datasetId, fail, notify, refreshJobs }) {
  const { job, stages, refresh } = useJob(jobId);
  const [drawer, setDrawer] = useState(null);
  if (!job || !stages) return <div className="empty-table small"><LoaderCircle className="spin" size={18} /></div>;
  const running = ["queued", "running"].includes(job.state);
  const combos = stages.combos || [];
  const ready = combos.flatMap((combo) => (combo.samples || []).filter((sample) => admissible(sample, datasetId)).map((sample) => sample.sample_id));
  const sequence = (stages.sequences || [])[0];
  const cancel = async () => {
    try { await post(`/api/v1/jobs/${jobId}/cancel`); await refresh(); await refreshJobs(); } catch (error) { fail(error); }
  };
  const rerun = async (comboIndex = null) => {
    try {
      await post(`/api/v1/jobs/${jobId}/continue`, { from_stage: "engine", combo_index: comboIndex, items: job.request });
      await refresh(); await refreshJobs();
      notify(comboIndex == null ? "整批重新生成" : "这一组重新生成");
    } catch (error) { fail(error); }
  };
  const admitAll = async () => { if (await admit(ready)) await refresh(); };
  return <div className={compact ? "job-detail compact" : "job-detail"}>
    <div className="job-detail-head">
      <span className={`state-chip ${job.state === "succeeded" ? "success" : running ? "running-state" : "warning"}`}>{running && <LoaderCircle size={12} className="spin" />}{jobLabel(job.state)}</span>
      <span>{job.completed}/{job.total} 组 · 通过 {job.kept} · 筛除 {job.dropped}</span>
      <div className="job-detail-actions">
        {running ? <button className="secondary-button" onClick={cancel}><X size={14} /> 取消</button> : job.request && <button className="secondary-button" onClick={() => rerun(null)}><RotateCcw size={14} /> 重跑</button>}
        <button className="secondary-button" disabled={!ready.length || !datasetId} title={datasetId ? "" : "先在顶栏选择数据集"} onClick={admitAll}>全部送审（{ready.length}）</button>
      </div>
    </div>
    {running && <div className="progress-track"><span style={{ width: `${Math.max(3, (job.completed / Math.max(job.total, 1)) * 100)}%` }} /></div>}
    {job.error && <p className="stage-note">{job.error}</p>}
    {sequence?.engine_message && <p className="stage-note">{sequence.engine_message}</p>}
    <div className={compact ? "combo-grid compact" : "combo-grid"}>{combos.map((combo) => {
      const samples = combo.samples || [];
      return <button type="button" key={combo.index} className="combo-card" onClick={() => setDrawer(combo.index)} disabled={!samples.length}>
        <div className="combo-shot">{combo.preview_url ? <img src={combo.preview_url} alt="" /> : <span className="combo-empty">{running ? "等待渲染" : "未产出"}</span>}</div>
        <div className="combo-meta"><small>{combo.scene} · {combo.weather} · {combo.viewpoint}{combo.camera_fallback ? "（借用侧视）" : ""}</small>{samples.length > 0 && <small>{combo.frames} 帧 · 通过 {combo.kept}{combo.dropped ? ` · 筛除 ${combo.dropped}` : ""}</small>}</div>
      </button>;
    })}</div>
    {drawer != null && <ComboDrawer combo={combos.find((item) => item.index === drawer)} job={job} datasetId={datasetId} onClose={() => setDrawer(null)} admit={async (ids) => { const ok = await admit(ids); if (ok) await refresh(); return ok; }} rerun={() => { setDrawer(null); rerun(drawer); }} />}
  </div>;
}

function ComboDrawer({ combo, job, datasetId, onClose, admit, rerun }) {
  const samples = combo?.samples || [];
  const [current, setCurrent] = useState(samples[0]?.sample_id || null);
  const [overlay, setOverlay] = useState(true);
  const [picked, setPicked] = useState([]);
  const [detail, setDetail] = useState(null);
  useEffect(() => {
    if (!current) return undefined;
    let alive = true;
    api(`/api/v1/samples/${encodeURIComponent(current)}`).then((data) => { if (alive) setDetail(data); }).catch(() => {});
    return () => { alive = false; };
  }, [current]);
  useEffect(() => {
    const close = (event) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [onClose]);
  if (!combo) return null;
  const choosable = samples.filter((sample) => admissible(sample, datasetId)).map((sample) => sample.sample_id);
  const toggle = (id) => setPicked((list) => (list.includes(id) ? list.filter((item) => item !== id) : [...list, id]));
  const boxes = (detail?.reviewed_objects || detail?.auto_objects || []);
  const running = ["queued", "running"].includes(job.state);
  return createPortal(<div className="drawer" role="dialog" aria-label="组合结果">
    <button className="scrim" type="button" aria-label="关闭" onClick={onClose} />
    <section className="drawer-sheet">
      <div className="surface-head"><div className="surface-title"><div><h2>{combo.scene} · {combo.weather} · {combo.viewpoint}</h2><p>{combo.frames} 帧 · {samples.length} 个样本</p></div></div><button className="icon-button" type="button" aria-label="关闭" onClick={onClose}><X size={15} /></button></div>
      <div className="drawer-view">
        {current && <div className="frame-stage">
          <img src={sampleFile(current, "image.png")} alt="样本画面" />
          {overlay && <img className="mask-overlay" src={sampleFile(current, "mask.png")} alt="" />}
          {overlay && <svg className="box-overlay" viewBox={`0 0 ${boxes[0]?.image_width || 256} ${boxes[0]?.image_height || 256}`} preserveAspectRatio="none">{boxes.map((obj, index) => <rect key={index} x={obj.bbox[0]} y={obj.bbox[1]} width={obj.bbox[2]} height={obj.bbox[3]} />)}</svg>}
        </div>}
        {combo.video_url && <video key={combo.video_url} src={combo.video_url} autoPlay loop muted playsInline controls aria-label="这一组的视频" />}
      </div>
      <div className="drawer-tools">
        <label className="check-chip"><input type="checkbox" checked={overlay} onChange={(event) => setOverlay(event.target.checked)} /><span>{overlay && <Check size={12} />}掩码和框</span></label>
        {detail && <small>{statusLabel(detail.status)}{detail.split ? ` · ${detail.split}` : ""} · {boxes.length} 个框 · {detail.quality?.checks?.reasons?.length ? `筛除原因：${detail.quality.checks.reasons.join("、")}` : `火焰占比 ${((detail.quality?.checks?.flame_area || 0) * 100).toFixed(2)}%`}{detail.quality?.checks?.flame_cut_by_edge ? " · 火焰被画面边缘截断" : ""}</small>}
      </div>
      <div className="frame-strip">{samples.map((sample) => <div key={sample.sample_id} className={sample.sample_id === current ? "frame-cell selected" : "frame-cell"}>
        <button type="button" onClick={() => setCurrent(sample.sample_id)} title={`第 ${sample.frame_index + 1} 帧${sample.augmentation ? ` · ${sample.augmentation}` : ""}`}><img src={sampleFile(sample.sample_id, "image.png")} alt="" loading="lazy" /></button>
        <small className={`status-${sample.status}`}>{sample.frame_index + 1}{sample.augmentation ? ` ${sample.augmentation}` : ""} · {statusLabel(sample.status)}</small>
        {admissible(sample, datasetId) && <input type="checkbox" aria-label="选择这一帧" checked={picked.includes(sample.sample_id)} onChange={() => toggle(sample.sample_id)} />}
      </div>)}</div>
      <div className="drawer-actions">
        <button className="secondary-button" type="button" disabled={!choosable.length} onClick={() => setPicked(picked.length === choosable.length ? [] : choosable)}>{picked.length === choosable.length && choosable.length ? "全不选" : "选全部可送审"}</button>
        <button className="primary-button" type="button" disabled={!picked.length || !datasetId} title={datasetId ? "" : "先在顶栏选择数据集"} onClick={async () => { if (await admit(picked)) setPicked([]); }}>送审 {picked.length} 个</button>
        <button className="secondary-button" type="button" disabled={running || !job.request} onClick={rerun}><RotateCcw size={14} /> 只重跑这一组</button>
      </div>
    </section>
  </div>, document.body);
}

function JobsPage({ jobs, jobId, datasetId, admit, fail, notify, refreshJobs, catalog }) {
  const selected = jobId || jobs[0]?.job_id || null;
  const job = jobs.find((item) => item.job_id === selected);
  const saveAsScene = async () => {
    try {
      const detail = await api(`/api/v1/jobs/${encodeURIComponent(selected)}`);
      const row = (detail.request || [])[0];
      if (!row) throw new Error("这个作业没有记录请求");
      const scene = await post("/api/v1/scenes", {
        name: `${row.scenario} · 来自作业 ${selected.slice(0, 6)}`,
        scenario: row.scenario,
        origin_fuel: row.fuel_type || "木材",
        placements: row.placements?.length ? row.placements : cleanPlacements(defaultPlacements(catalog, row.scenario)),
        origin: row.origin || null,
        cameras: row.cameras || {},
        gas_domain: row.gas_domain || null,
        lights: row.lights ?? null,
        simulation: row.simulation || null,
      });
      go("scenes", scene.scene_id);
    } catch (error) { fail(error); }
  };
  return <>
    <div className="page-heading"><div><div className="eyebrow">JOBS</div><h1>作业</h1><p>每个作业烘焙一次火，按组合渲染；每一帧是一个样本。点一组看帧、掩码和视频，选中的帧送去标注。</p></div><button className="secondary-button" onClick={() => refreshJobs().catch(fail)}><RefreshCw size={15} /> 刷新</button></div>
    {!jobs.length ? <EmptyState icon={Activity} title="还没有作业" message="在场景库打开一个场景，在右侧生成。" /> : <div className="jobs-layout">
      <nav className="surface job-list" aria-label="作业列表">{jobs.map((item) => <a key={item.job_id} href={`#/jobs/${item.job_id}`} className={item.job_id === selected ? "job-row selected" : "job-row"}>
        <StateDot state={item.state} />
        <span className="job-row-copy"><strong>{item.scenario || "作业"} · #{item.job_id.slice(0, 6)}</strong><small>{formatDate(item.created_at)} · {item.completed}/{item.total} 组 · 通过 {item.kept}</small></span>
      </a>)}</nav>
      <section className="surface job-panel">
        {job && <div className="job-panel-head"><h2>#{job.job_id.slice(0, 10)}</h2>{job.scene_ids?.length ? <a className="secondary-button" href={`#/scenes/${job.scene_ids[0]}`}><FolderOpen size={14} /> 打开场景</a> : <button className="secondary-button" onClick={saveAsScene}><Save size={14} /> 摆放存为场景</button>}</div>}
        {selected && <JobDetail key={selected} jobId={selected} admit={admit} datasetId={datasetId} fail={fail} notify={notify} refreshJobs={refreshJobs} />}
      </section>
    </div>}
  </>;
}

function DatasetsPage({ catalog, datasets, datasetId, setDatasetId, refreshDatasets, fail, notify }) {
  const [name, setName] = useState("");
  const [versions, setVersions] = useState([]);
  const [head, setHead] = useState(null);
  const [selected, setSelected] = useState(null);
  const [files, setFiles] = useState([]);
  const [compareIds, setCompareIds] = useState([]);
  const [comparison, setComparison] = useState(null);
  const [syncState, setSyncState] = useState("");
  const [publishing, setPublishing] = useState(false);
  const dataset = datasets.find((item) => item.dataset_id === datasetId);
  const refreshVersions = useCallback(async () => {
    if (!datasetId) { setVersions([]); setHead(null); return; }
    const data = await api(`/api/v1/versions?dataset_id=${encodeURIComponent(datasetId)}&limit=200`);
    setVersions(data.versions || []); setHead(data.head);
  }, [datasetId]);
  const sync = useCallback(async (quiet = false) => {
    try {
      const result = await post("/api/v1/samples/sync-reviews");
      if (!result.label_studio?.success) { setSyncState(`Label Studio 连接失败：${result.label_studio?.error || "未知原因"}`); return; }
      setSyncState(`${new Date().toLocaleTimeString("zh-CN")} 已同步 ${result.label_studio.count} 个任务${result.updated.length ? `，回写 ${result.updated.length} 条` : ""}${result.frozen.length ? `，${result.frozen.length} 条已发布不改` : ""}`);
      if (result.updated.length) { await refreshDatasets(); if (!quiet) notify(`回写 ${result.updated.length} 条审核`); }
    } catch (error) { if (!quiet) fail(error); }
  }, [fail, notify, refreshDatasets]);
  useEffect(() => { setSelected(null); setFiles([]); setCompareIds([]); refreshVersions().catch(fail); }, [fail, refreshVersions]);
  useEffect(() => {
    sync(true);
    const timer = window.setInterval(() => sync(true), 30000);
    const focus = () => sync(true);
    window.addEventListener("focus", focus);
    return () => { window.clearInterval(timer); window.removeEventListener("focus", focus); };
  }, [sync]);
  useEffect(() => {
    if (compareIds.length !== 2) { setComparison(null); return; }
    api(`/api/v1/versions/compare?base=${encodeURIComponent(compareIds[0])}&other=${encodeURIComponent(compareIds[1])}`).then(setComparison).catch(fail);
  }, [compareIds, fail]);
  const create = async (event) => {
    event.preventDefault();
    if (!name.trim()) return;
    try { const created = await post("/api/v1/datasets", { name: name.trim() }); await refreshDatasets(); setDatasetId(created.dataset_id); setName(""); } catch (error) { fail(error); }
  };
  const publish = async () => {
    if (!dataset) return;
    setPublishing(true);
    try {
      const record = await post(`/api/v1/datasets/${encodeURIComponent(datasetId)}/publish`);
      await refreshDatasets(); await refreshVersions();
      notify(`已发布 v${record["版本号"]}`);
      open(record);
    } catch (error) { fail(error); } finally { setPublishing(false); }
  };
  const open = async (record) => {
    setSelected(record);
    try { setFiles((await api(`/api/v1/versions/${encodeURIComponent(record.version_id)}/files`)).files || []); } catch (error) { fail(error); }
  };
  const rollback = async (record) => {
    if (!window.confirm(`把 v${record["版本号"]} 设为当前版本？下次发布以它为基础，之后版本新增的样本不会自动带上。`)) return;
    try { await post(`/api/v1/versions/${encodeURIComponent(record.version_id)}/rollback`); await refreshDatasets(); await refreshVersions(); notify(`当前版本改为 v${record["版本号"]}`); } catch (error) { fail(error); }
  };
  const toggleCompare = (id) => setCompareIds((list) => (list.includes(id) ? list.filter((item) => item !== id) : [...list.slice(-1), id]));
  const counts = dataset?.status_counts || {};
  const byId = Object.fromEntries(versions.map((item) => [item.version_id, item]));
  return <>
    <div className="page-heading"><div><div className="eyebrow">DATASETS</div><h1>数据集与版本</h1><p>每个版本包含数据集里全部已审核样本，记录父版本。同一段序列的帧固定在同一个划分里，导出可以直接训练。</p></div></div>
    <div className="surface dataset-head">
      <label className="field"><span className="field-label">数据集</span><select value={datasetId} onChange={(event) => setDatasetId(event.target.value)}><option value="">选择数据集</option>{datasets.map((item) => <option key={item.dataset_id} value={item.dataset_id}>{item.name}</option>)}</select></label>
      <form className="dataset-create" onSubmit={create}><input value={name} maxLength={80} placeholder="新数据集名称" onChange={(event) => setName(event.target.value)} /><button className="secondary-button" type="submit"><Plus size={14} /> 新建</button></form>
    </div>
    {!dataset ? <EmptyState icon={Database} title="先选一个数据集" message="送审和发布都在当前数据集里进行，顶栏也可以切换。" /> : <>
      <div className="summary-row">
        <Summary label="可送审" value={counts.generated || 0} />
        <Summary label="审核中" value={counts.pending_review || 0} />
        <Summary label="已审核" value={counts.reviewed || 0} />
        <Summary label="已拒绝" value={counts.rejected || 0} />
      </div>
      <div className="surface publish-bar">
        <div><strong>{dataset.head_revision ? `当前 v${dataset.head_revision}` : "还没有版本"}</strong><small>{dataset.pending_added || dataset.pending_removed ? `发布后：+${dataset.pending_added} / −${dataset.pending_removed}，共 ${dataset.ready_to_publish} 个样本` : "和当前版本相比没有变化"}</small></div>
        <small className="sync-state">{syncState || "正在同步 Label Studio"}</small>
        <button className="secondary-button" onClick={() => sync(false)}><RefreshCw size={14} /> 同步审核</button>
        <a className="secondary-button" href="/auth/label-studio" target="_blank" rel="noreferrer">打开 Label Studio <ArrowUpRight size={13} /></a>
        <button className="primary-button" disabled={!dataset.ready_to_publish || publishing} onClick={publish}>{publishing ? <LoaderCircle size={15} className="spin" /> : <Database size={15} />}发布新版本</button>
      </div>
      <SdAugmentPanel catalog={catalog} datasetId={datasetId} reviewed={counts.reviewed || 0} refreshDatasets={refreshDatasets} fail={fail} notify={notify} />
      {comparison && <ComparePanel comparison={comparison} />}
      <div className="version-layout">
        <section className="surface table-surface">
          <div className="table-head version-table-head"><span>版本</span><span>样本</span><span>变化</span><span>划分</span><span>操作</span></div>
          {!versions.length ? <EmptyState icon={Database} title="暂无版本" message="样本审核完后发布第一个版本。" /> : versions.map((record) => {
            const meta = record.metadata || {};
            const volume = meta["数据统计"]?.["数据量"] || {};
            const split = meta["数据统计"]?.["划分"] || {};
            const change = meta["变化"] || {};
            const parent = record.parent_id ? byId[record.parent_id] : null;
            return <div className={selected?.version_id === record.version_id ? "version-row selected" : "version-row"} key={record.version_id}>
              <button className="version-select" onClick={() => open(record)}><strong>v{record["版本号"]}</strong><small>{parent ? `基于 v${parent["版本号"]}` : "第一个版本"} · {formatDate(record.created_at)}</small>{head === record.version_id && <i>当前</i>}</button>
              <span className="version-count">{volume.samples ?? 0} 个样本<small>{volume.boxes ?? 0} 个框 · {volume.sequences ?? 0} 段序列</small></span>
              <span className="version-error">+{change.added ?? volume.samples ?? 0} / −{change.removed ?? 0}</span>
              <span className="version-quality">{["train", "val", "test"].map((name) => `${name} ${split[name]?.count ?? 0}`).join(" · ")}</span>
              <div className="row-actions">
                <button className={compareIds.includes(record.version_id) ? "icon-button selected" : "icon-button"} onClick={() => toggleCompare(record.version_id)} title="加入对比" aria-label="加入对比"><Check size={15} /></button>
                <a className="icon-button" href={`/api/v1/versions/${encodeURIComponent(record.version_id)}/archive`} title="下载 ZIP" aria-label="下载 ZIP"><ArrowDownToLine size={15} /></a>
                <button className="icon-button" disabled={head === record.version_id} onClick={() => rollback(record)} title="设为当前版本" aria-label="设为当前版本"><RotateCcw size={15} /></button>
              </div>
            </div>;
          })}
        </section>
        <section className="surface version-detail-surface">
          {selected ? <>
            <div className="surface-head"><div className="surface-title"><div><h2>v{selected["版本号"]}</h2><p>{files.length} 个文件 · 质量 {selected.metadata?.["质量报告"]?.fid == null ? "FID 未计算" : `FID ${Number(selected.metadata["质量报告"].fid).toFixed(2)}`}{selected.metadata?.["质量报告"]?.is == null ? "" : ` · IS ${Number(selected.metadata["质量报告"].is).toFixed(2)}`}</p></div></div></div>
            <div className="detail-id">{selected.version_id}</div>
            <div className="file-list">{files.filter((file) => !file.path.startsWith("images/") && !file.path.startsWith("labels/") && !file.path.startsWith("annotations/voc/")).map((file) => <a key={file.path} href={versionFileUrl(selected.version_id, file.path)} target="_blank" rel="noreferrer"><FileText size={14} /><span>{file.path}</span><small>{formatBytes(file.size)}</small><ArrowUpRight size={13} /></a>)}</div>
            <p className="field-hint">另有 {files.filter((file) => file.path.startsWith("images/")).length} 张图像和对应的 YOLO / VOC 标注，下载 ZIP 获取全部。</p>
          </> : <EmptyState icon={FileText} title="选一个版本" message="查看文件和质量报告。勾两个版本对比样本差异。" />}
        </section>
      </div>
    </>}
  </>;
}

const SD_DEFAULTS = { variants: 1, styles: [], strength: 0.2, steps: 20, cfg_scale: 7 };

/** SD restyles the background of reviewed samples. Children keep the boxes and are reviewed already. */
function SdAugmentPanel({ catalog, datasetId, reviewed, refreshDatasets, fail, notify }) {
  const [state, setState] = useState({ eligible: 0, jobs: [] });
  const [form, setForm] = useState(SD_DEFAULTS);
  const [busy, setBusy] = useState(false);
  const refresh = useCallback(async () => {
    if (!datasetId) return null;
    const next = await api(`/api/v1/datasets/${encodeURIComponent(datasetId)}/sd-augment`);
    setState(next);
    return next;
  }, [datasetId]);
  useEffect(() => { refresh().catch(fail); }, [refresh, fail, reviewed]);
  const running = state.jobs.find((job) => job.state === "queued" || job.state === "running");
  useEffect(() => {
    if (!running) return undefined;
    const timer = window.setInterval(async () => {
      try {
        const next = await refresh();
        const job = next?.jobs.find((item) => item.job_id === running.job_id);
        if (job && job.state !== "queued" && job.state !== "running") {
          await refreshDatasets();
          notify(`SD 增强${jobLabel(job.state)}：新增 ${job.created} 个已审核样本${job.failed ? `，${job.failed} 个失败` : ""}`);
        }
      } catch (error) { fail(error); }
    }, 2000);
    return () => window.clearInterval(timer);
  }, [running, refresh, refreshDatasets, fail, notify]);
  const styles = Object.entries(catalog.sd_styles || {}).flatMap(([environment, items]) => items.map((item) => ({ ...item, environment })));
  const toggleStyle = (key) => setForm((current) => ({ ...current, styles: current.styles.includes(key) ? current.styles.filter((item) => item !== key) : [...current.styles, key] }));
  const submit = async () => {
    setBusy(true);
    try {
      await post(`/api/v1/datasets/${encodeURIComponent(datasetId)}/sd-augment`, {
        variants: Number(form.variants),
        styles: form.styles.length ? form.styles : null,
        strength: Number(form.strength),
        steps: Number(form.steps),
        cfg_scale: Number(form.cfg_scale),
      });
      await refresh();
    } catch (error) { fail(error); } finally { setBusy(false); }
  };
  const cancel = async () => {
    try { await post(`/api/v1/sd-augment/${encodeURIComponent(running.job_id)}/cancel`); await refresh(); } catch (error) { fail(error); }
  };
  const last = state.jobs[0];
  return <section className="surface sd-panel">
    <div className="sd-panel-head">
      <div><strong>SD 增强</strong><small>只对已审核的原始样本重绘背景，火焰像素不变，派生样本直接带审核过的框，和原样本同一划分。</small></div>
      <span className="sd-eligible">{state.eligible} 个可增强</span>
    </div>
    <div className="sd-panel-body">
      <Field label="每个样本"><select value={form.variants} onChange={(event) => setForm({ ...form, variants: event.target.value })}>{[1, 2, 3, 4].map((n) => <option key={n} value={n}>{n} 份</option>)}</select></Field>
      <Field label="强度"><input type="number" min="0.05" max="0.6" step="0.05" value={form.strength} onChange={(event) => setForm({ ...form, strength: event.target.value })} /></Field>
      <Field label="步数"><input type="number" min="1" max="20" step="1" value={form.steps} onChange={(event) => setForm({ ...form, steps: event.target.value })} /></Field>
      <Field label="CFG"><input type="number" min="0" max="20" step="0.5" value={form.cfg_scale} onChange={(event) => setForm({ ...form, cfg_scale: event.target.value })} /></Field>
    </div>
    {styles.length > 0 && <div className="choice-block"><span className="field-label">背景风格（不选则按样本的室内/室外轮换全部）</span><div className="choice-row">{styles.map((item) => <label className={form.styles.includes(item.key) ? "check-chip checked" : "check-chip"} key={item.key}><input type="checkbox" checked={form.styles.includes(item.key)} onChange={() => toggleStyle(item.key)} /><span>{form.styles.includes(item.key) && <Check size={12} />}{item.environment} · {item.label}</span></label>)}</div></div>}
    <div className="sd-panel-foot">
      {last && <small>{running ? `进行中 ${last.completed}/${last.total}` : `上次${jobLabel(last.state)}：${last.completed}/${last.total} 个样本，新增 ${last.created}${last.failed ? `，失败 ${last.failed}` : ""}`}{last.error && !running ? ` · ${last.error.slice(0, 120)}` : ""}</small>}
      {running
        ? <button className="secondary-button" onClick={cancel}><X size={14} /> 取消</button>
        : <button className="primary-button" disabled={!state.eligible || busy} onClick={submit}>{busy ? <LoaderCircle size={15} className="spin" /> : <Layers3 size={15} />}增强 {state.eligible} 个样本</button>}
    </div>
  </section>;
}

function ComparePanel({ comparison }) {
  const volume = comparison.data_volume;
  const diff = comparison.samples?.counts || {};
  return <section className="surface compare-panel"><div className="surface-head"><div className="surface-title"><div><h2>v{comparison.base?.["版本号"]} → v{comparison.other?.["版本号"]}</h2><p>新增 {diff.added || 0} · 移除 {diff.removed || 0} · 换划分 {diff.split_changed || 0}</p></div></div></div><div className="compare-grid">
    <article className="compare-card"><span>数据量</span><strong>{volume.delta.samples >= 0 ? "+" : ""}{volume.delta.samples} 样本</strong><small>{volume.base.samples} → {volume.other.samples} 样本 · {volume.base.boxes} → {volume.other.boxes} 框</small></article>
    <article className="compare-card"><span>天气</span><strong>占比</strong><small>{ratioText(comparison.scene_weather?.base?.["天气比例"])}<br />{ratioText(comparison.scene_weather?.other?.["天气比例"])}</small></article>
    <article className="compare-card"><span>标注错误率</span><strong>{(Number(comparison.annotation_error_rate.delta) * 100).toFixed(2)} 个百分点</strong><small>{(Number(comparison.annotation_error_rate.base) * 100).toFixed(2)}% → {(Number(comparison.annotation_error_rate.other) * 100).toFixed(2)}%</small></article>
    <article className="compare-card"><span>新增样本</span><strong>{diff.added || 0}</strong><small className="id-list">{(comparison.samples?.added || []).slice(0, 6).map((id) => id.slice(0, 10)).join(" · ") || "-"}</small></article>
  </div></section>;
}

function ChoiceGroup({ label, options, selected, onToggle, disabled = [] }) {
  return <div className="choice-block"><span className="field-label">{label}</span><div className="choice-row" role="group" aria-label={label}>{(options || []).map((name) => {
    const off = disabled.includes(name);
    const on = selected.includes(name) && !off;
    return <label className={`check-chip${on ? " checked" : ""}${off ? " disabled" : ""}`} key={name}><input type="checkbox" checked={on} disabled={off} onChange={() => onToggle(name)} /><span>{on && <Check size={12} />}{name}</span></label>;
  })}</div></div>;
}
function Field({ label, unit, children }) {
  return <label className="field"><span className="field-label">{label}{unit && <i>{unit}</i>}</span>{children}</label>;
}
function Summary({ label, value }) {
  return <div className="summary-item"><div><small>{label}</small><strong>{Number(value).toLocaleString()}</strong></div></div>;
}
function EmptyState({ icon: Icon, title, message }) {
  return <div className="empty-table"><Icon size={22} /><strong>{title}</strong>{message && <span>{message}</span>}</div>;
}

function LoginScreen({ session, onAuthenticated }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(session.error || "");
  const [busy, setBusy] = useState(false);
  const creating = Boolean(session.needs_bootstrap);
  const submit = async (event) => {
    event.preventDefault();
    setBusy(true); setError("");
    try {
      onAuthenticated(await post(creating ? "/api/v1/auth/bootstrap" : "/api/v1/auth/login", { email, password }));
    } catch (submitError) { setError(submitError.message); } finally { setBusy(false); }
  };
  return <main className="login-screen"><form className="login-card" onSubmit={submit}><a className="brand" href="/" aria-label="FirePhantom 首页"><span className="brand-mark"><Flame size={18} strokeWidth={2.4} /></span><span>FirePhantom<small>DATASET LAB</small></span></a><h1>{creating ? "创建管理员" : "登录工作台"}</h1><p>{creating ? "第一个账号是管理员。之后从这里打开标注页，不需要再登录 Label Studio。" : "使用 FirePhantom 账号。打开标注页时不会再看到 Label Studio 登录框。"}</p><Field label="邮箱"><input type="email" required value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="username" /></Field><Field label="密码"><input type="password" required minLength={8} value={password} onChange={(event) => setPassword(event.target.value)} autoComplete={creating ? "new-password" : "current-password"} /></Field>{error && <div className="login-error">{error}</div>}<button className="primary-button" type="submit" disabled={busy}>{busy ? "请稍候" : creating ? "创建并进入" : "登录"}</button></form></main>;
}

function Root() {
  const [session, setSession] = useState(undefined);
  useEffect(() => {
    api("/api/v1/auth/me").then(setSession).catch((error) => setSession({ authenticated: false, needs_bootstrap: false, error: error.message }));
  }, []);
  if (!session) return <main className="login-screen"><p>正在连接工作台…</p></main>;
  if (!session.authenticated) return <LoginScreen session={session} onAuthenticated={setSession} />;
  return <App session={session} onLogout={() => setSession({ authenticated: false, needs_bootstrap: false })} />;
}

createRoot(document.getElementById("root")).render(<Root />);
