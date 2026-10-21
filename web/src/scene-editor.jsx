import React, { Suspense, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { GizmoHelper, GizmoViewport, OrbitControls, TransformControls, useGLTF } from "@react-three/drei";
import { Box, ChevronDown, ChevronRight, Copy, Cuboid, Eye, EyeOff, Flame, Focus, Globe, Lightbulb, Move3d, Plus, Rotate3d, RotateCcw, Scaling, Square, Trash2, TriangleAlert, Video } from "lucide-react";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RectAreaLightUniformsLib } from "three/addons/lights/RectAreaLightUniformsLib.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import * as THREE from "three";

RectAreaLightUniformsLib.init();

// Scene meters are Z-up, matching Blender. Three.js is Y-up.
// One conversion, the same one as firephantom/physics/placements.py:
//   (three.x, three.y, three.z) = (x, z, -y)
//   (x, y, z) = (three.x, -three.z, three.y)
// Scene yaw about +Z equals rotation.y. rotation.y = π/2 sends local +X to (0, 0, -1).

export function sceneToThree(x, y, z) {
  return [x, z, -y];
}

export function threeToScene(x, y, z) {
  return [x, -z, y];
}

const flagWorld = new THREE.Vector3();
const flagCamera = new THREE.Vector3();
const flagDir = new THREE.Vector3();

// drei's Html mounts a second React root and unmounts it during the outer
// commit. React 19 then calls removeChild on a node that has already moved.
// These labels are one span, so they stay plain DOM.
function SceneFlag({ position, label, picked = false }) {
  const group = useRef(null);
  const { camera, gl, size } = useThree();
  const nodes = useRef(null);
  useLayoutEffect(() => {
    const parent = gl.domElement.parentNode;
    if (!parent) return undefined;
    const el = document.createElement("div");
    el.className = "package-flag-wrap";
    el.style.position = "absolute";
    el.style.top = "0";
    el.style.left = "0";
    const span = document.createElement("span");
    el.appendChild(span);
    parent.appendChild(el);
    nodes.current = { el, span };
    return () => {
      nodes.current = null;
      el.remove();
    };
  }, [gl]);
  useLayoutEffect(() => {
    const current = nodes.current;
    if (!current) return;
    current.span.textContent = label;
    current.span.className = picked ? "package-flag picked" : "package-flag";
  }, [label, picked]);
  useFrame(() => {
    const object = group.current;
    const current = nodes.current;
    if (!object || !current) return;
    const show = (display, transform, depth) => {
      if (current.display !== display) {
        current.display = display;
        current.el.style.display = display;
      }
      if (display || current.transform === transform) return;
      current.transform = transform;
      current.el.style.transform = transform;
      current.el.style.zIndex = depth;
    };
    object.updateWorldMatrix(true, false);
    flagWorld.setFromMatrixPosition(object.matrixWorld);
    camera.getWorldDirection(flagDir);
    flagCamera.setFromMatrixPosition(camera.matrixWorld);
    const facing = (flagWorld.x - flagCamera.x) * flagDir.x
      + (flagWorld.y - flagCamera.y) * flagDir.y
      + (flagWorld.z - flagCamera.z) * flagDir.z;
    if (facing <= 0) {
      show("none");
      return;
    }
    flagWorld.project(camera);
    const x = Math.round((flagWorld.x * 0.5 + 0.5) * size.width);
    const y = Math.round((-flagWorld.y * 0.5 + 0.5) * size.height);
    show("", `translate(${x}px, ${y}px) translate(-50%, -50%)`, String(Math.round((1 - flagWorld.z) * 20)));
  });
  return <group ref={group} position={position} />;
}

const FLOOR = {
  房间火灾: "#8d7352",
  林区火灾: "#3e4c34",
  工业区火灾: "#6e7472",
};
const FUEL_COLOR = {
  木材: "#c4894e",
  植被: "#6d8a4e",
  塑料: "#7f97a1",
  汽油: "#4a4036",
  化学品: "#8a7a48",
};
const ISSUE_LABEL = {
  shaft: "挡住火柱",
  outside: "超出场景",
  far: "气体域盖不住",
};
const LIGHT_LABEL = { POINT: "点光", SUN: "太阳", SPOT: "聚光", AREA: "面光" };
const SHELL_LABEL = {
  floor: "地板",
  wall_west: "西墙",
  wall_east_south: "东墙南段",
  wall_east_north: "东墙北段",
  wall_east_lintel: "东墙门梁",
  wall_south: "南墙",
  wall_north: "北墙",
  ceiling: "天花板",
  fuel_bed: "燃料床",
  canopy: "树冠",
  tank: "罐体",
  pipe_rack: "管架",
  pool: "油池",
};
const KIND_LABEL = { ignition: "起火面", fuel: "可燃", inert: "不燃烧" };
const STRUCTURE_ROLES = ["inert", "fuel", "ignition"];

function round4(value) {
  return Math.round(Number(value) * 10000) / 10000;
}

/** Preset obstacles in the stored shape. Null structures on a scene means this list. */
export function boxesFromObstacles(shell) {
  return (shell?.obstacles || []).map((obstacle) => {
    const xb = obstacle.xb || [];
    const center = [0, 1, 2].map((axis) => round4((Number(xb[axis * 2]) + Number(xb[axis * 2 + 1])) / 2));
    const size = [0, 1, 2].map((axis) => round4(Number(xb[axis * 2 + 1]) - Number(xb[axis * 2])));
    const id = obstacle.name;
    return {
      id,
      name: SHELL_LABEL[id] || obstacle.label || id,
      role: obstacle.kind || obstacle.role || "inert",
      center,
      size,
    };
  });
}

/** One stored box as the obstacle record the viewport and the shaft cut already use. */
export function boxToObstacle(box) {
  const [cx, cy, cz] = box.center;
  const [dx, dy, dz] = box.size;
  return {
    name: box.id,
    label: box.name,
    kind: box.role || "inert",
    center: box.center,
    size: box.size,
    xb: [cx - dx / 2, cx + dx / 2, cy - dy / 2, cy + dy / 2, cz - dz / 2, cz + dz / 2],
  };
}

export function structureInside(box, domain) {
  if (!box?.center || !box?.size || !domain) return false;
  const [length, width, height] = domain;
  const [cx, cy, cz] = box.center;
  const [dx, dy, dz] = box.size;
  const eps = 1e-4;
  return cx - dx / 2 >= -eps && cy - dy / 2 >= -eps && cz - dz / 2 >= -eps
    && cx + dx / 2 <= length + eps && cy + dy / 2 <= width + eps && cz + dz / 2 <= height + eps
    && dx <= length + eps && dy <= width + eps && dz <= height + eps
    && dx >= 0.05 - eps && dy >= 0.05 - eps && dz >= 0.05 - eps;
}

/** Pull a box back inside the site. Edges stay between 0.05 m and the matching site edge. */
export function clampStructure(box, domain) {
  const limits = domain || [1, 1, 1];
  const size = [0, 1, 2].map((index) => {
    let edge = Number(box.size?.[index]);
    if (!Number.isFinite(edge)) edge = 0.2;
    return round4(Math.min(limits[index], Math.max(0.05, edge)));
  });
  const center = [0, 1, 2].map((index) => {
    let value = Number(box.center?.[index]);
    const half = size[index] / 2;
    if (!Number.isFinite(value)) value = half;
    return round4(Math.min(limits[index] - half, Math.max(half, value)));
  });
  return { ...box, center, size };
}

function sameStructure(left, right) {
  return left.id === right.id && left.name === right.name && left.role === right.role
    && sameVec(left.center, right.center) && sameVec(left.size, right.size);
}

/** A list that still matches the preset is stored as null. An empty list stays empty. */
export function collapseStructures(preset, rows) {
  if (rows == null) return null;
  const order = (list) => [...list].sort((left, right) => (left.id < right.id ? -1 : left.id > right.id ? 1 : 0));
  const given = order(rows);
  const base = order(preset || []);
  if (given.length === base.length && given.every((item, index) => sameStructure(item, base[index]))) return null;
  return rows;
}

export function nextStructureId(boxes) {
  const used = new Set((boxes || []).map((box) => box.id));
  let index = 1;
  while (used.has(`s${index}`)) index += 1;
  return `s${index}`;
}

function isCeiling(obstacle) {
  return obstacle?.name === "ceiling" || obstacle?.label === "天花板";
}

/** Eye state. A ceiling with no key is hidden; hidden[key] === false forces it on. */
export function shellEyeOn(obstacle, hidden) {
  const value = hidden?.[`shell:${obstacle?.name}`];
  if (value === true) return false;
  if (isCeiling(obstacle)) return value === false;
  return true;
}

/** Drawn when the eye is on, or while it is selected, unless the eye forced it off. */
export function shellDrawn(obstacle, hidden, selected) {
  if (hidden?.[`shell:${obstacle?.name}`] === true) return false;
  if (selected?.kind === "shell" && selected.id === obstacle?.name) return true;
  return shellEyeOn(obstacle, hidden);
}

/** Catalog shell plus the scene's site, boxes, and fire point. */
export function liveLayout(shell, { structures = null, domain = null, origin = null } = {}) {
  if (!shell) return null;
  const site = Array.isArray(domain) && domain.length === 3 ? domain : shell.domain;
  const boxes = structures == null ? null : structures;
  return {
    ...shell,
    domain: site,
    origin: origin || shell.origin,
    obstacles: boxes == null ? shell.obstacles : boxes.map(boxToObstacle),
  };
}

export function defaultPlacements(catalog, scenario) {
  const defaults = catalog?.defaults?.[scenario];
  return Array.isArray(defaults) ? defaults.map((entry) => ({ ...entry, size: Array.isArray(entry.size) ? [...entry.size] : entry.size })) : [];
}

export function yawedBox(item) {
  const hx = item.size[0] / 2;
  const hy = item.size[1] / 2;
  const hz = item.size[2] / 2;
  const cosine = Math.cos(item.yaw || 0);
  const sine = Math.sin(item.yaw || 0);
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const x of [-hx, hx]) {
    for (const y of [-hy, hy]) {
      const worldX = item.x + cosine * x - sine * y;
      const worldY = item.y + sine * x + cosine * y;
      minX = Math.min(minX, worldX);
      maxX = Math.max(maxX, worldX);
      minY = Math.min(minY, worldY);
      maxY = Math.max(maxY, worldY);
    }
  }
  return [minX, maxX, minY, maxY, item.z - hz, item.z + hz];
}

function overlaps(a, b) {
  return a[0] < b[1] && b[0] < a[1] && a[2] < b[3] && b[2] < a[3] && a[4] < b[5] && b[4] < a[5];
}

export function farFromFire(item, items, origin, cap) {
  const points = [[origin[0], origin[1]]];
  for (const other of items) {
    if (other.ignition) points.push([other.x, other.y]);
  }
  let best = Infinity;
  for (const [gx, gy] of points) best = Math.min(best, Math.hypot(item.x - gx, item.y - gy));
  return best > cap;
}

export function insideDomain(item, domain) {
  if (!domain || !item?.size) return true;
  const [minX, maxX, minY, maxY, minZ, maxZ] = yawedBox(item);
  const [length, width, height] = domain;
  const eps = 1e-4;
  return minX >= -eps && maxX <= length + eps && minY >= -eps && maxY <= width + eps && minZ >= -eps && maxZ <= height + eps;
}

/** Floor-to-top gap the bake cuts out of every solid around the flame. */
export function shaftBox(shell) {
  const half = shell?.shaft_half_m || 1;
  const [ox, oy] = shell.origin;
  return [ox - half, ox + half, oy - half, oy + half, 0, shell.domain[2]];
}

/** True when a point sits inside a solid after the flame shaft is cut out. */
export function pointInSolid(point, shell) {
  if (!point || !shell?.origin || !shell?.domain || !shell?.obstacles?.length) return false;
  const [x, y, z] = point.map(Number);
  if (![x, y, z].every(Number.isFinite)) return false;
  const shaft = shaftBox(shell);
  return shell.obstacles.some((obstacle) => obstacle?.xb && subtractBox(obstacle.xb, shaft).some((xb) => (
    x > xb[0] && x < xb[1] && y > xb[2] && y < xb[3] && z > xb[4] && z < xb[5]
  )));
}

/** Box minus shaft, piece for piece as blender_scene._subtract_box builds it. */
export function subtractBox(box, shaft) {
  const [ox0, ox1, oy0, oy1, oz0, oz1] = box;
  const [sx0, sx1, sy0, sy1, sz0, sz1] = shaft;
  const ix0 = Math.max(ox0, sx0), ix1 = Math.min(ox1, sx1);
  const iy0 = Math.max(oy0, sy0), iy1 = Math.min(oy1, sy1);
  const iz0 = Math.max(oz0, sz0), iz1 = Math.min(oz1, sz1);
  const eps = 1e-6;
  if (ix0 >= ix1 - eps || iy0 >= iy1 - eps || iz0 >= iz1 - eps) return [box];
  const pieces = [];
  if (oz0 < iz0 - eps) pieces.push([ox0, ox1, oy0, oy1, oz0, iz0]);
  if (iz1 < oz1 - eps) pieces.push([ox0, ox1, oy0, oy1, iz1, oz1]);
  if (ox0 < ix0 - eps) pieces.push([ox0, ix0, oy0, oy1, iz0, iz1]);
  if (ix1 < ox1 - eps) pieces.push([ix1, ox1, oy0, oy1, iz0, iz1]);
  if (oy0 < iy0 - eps) pieces.push([ix0, ix1, oy0, iy0, iz0, iz1]);
  if (iy1 < oy1 - eps) pieces.push([ix0, ix1, iy1, oy1, iz0, iz1]);
  return pieces;
}

function blocksShaft(item, shell) {
  return overlaps(yawedBox(item), shaftBox(shell));
}

export function boundsOfGas(gas) {
  if (!gas?.location || !gas?.scale || gas.location.length !== 3 || gas.scale.length !== 3) return null;
  const [x, y, z] = gas.location;
  const [sx, sy, sz] = gas.scale;
  return [x - sx, x + sx, y - sy, y + sy, z - sz, z + sz];
}

export function sameVec(left, right, tol = 1e-3) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((value, index) => Math.abs(Number(value) - Number(right[index])) <= tol);
}

export function samePose(left, right) {
  if (!left || !right) return false;
  return sameVec(left.location, right.location) && sameVec(left.target, right.target) && Math.abs(Number(left.focal_mm) - Number(right.focal_mm)) <= 0.05;
}

export function sameGas(left, right) {
  if (!left || !right) return false;
  return sameVec(left.location, right.location) && sameVec(left.scale, right.scale) && Number(left.resolution) === Number(right.resolution);
}

export function clampOrigin(point, domain) {
  const [length, width, height] = domain;
  const margin = 0.2;
  return [
    Math.min(length - margin, Math.max(margin, point[0])),
    Math.min(width - margin, Math.max(margin, point[1])),
    Math.min(height - 0.05, Math.max(0, point[2])),
  ];
}

export function clampCameraPoint(point, domain) {
  const [length, width, height] = domain;
  const margin = 0.05;
  return [
    Math.min(length - margin, Math.max(margin, point[0])),
    Math.min(width - margin, Math.max(margin, point[1])),
    Math.min(height - margin, Math.max(margin, point[2])),
  ];
}

/**
 * Keep a custom gas box inside the scene, at least 0.5 m a side. Its size has
 * no cap of its own; the bake budget from the preview limits it instead.
 */
export function clampGas(gas, domain, maxResolution = 128) {
  const minHalf = 0.25;
  const [length, width, height] = domain;
  const maxHalf = [length / 2, width / 2, height / 2];
  const scale = [0, 1, 2].map((index) => Math.min(maxHalf[index], Math.max(minHalf, Number(gas.scale?.[index]) || minHalf)));
  const [sx, sy, sz] = scale;
  const location = [
    Math.min(length - sx, Math.max(sx, Number(gas.location?.[0]) || sx)),
    Math.min(width - sy, Math.max(sy, Number(gas.location?.[1]) || sy)),
    Math.min(height - sz, Math.max(sz, Number(gas.location?.[2]) || sz)),
  ];
  let resolution = Math.round(Number(gas.resolution));
  if (!Number.isFinite(resolution)) resolution = 32;
  resolution = Math.min(Number(maxResolution) || 128, Math.max(16, resolution));
  return { location, scale, resolution };
}

export const GAS_STEPS = [[24, "粗"], [32, "标准"], [48, "细"], [64, "极细"]];
export const NOISE_STEPS = [[0, "关"], [2, "×2"], [3, "×3"]];

/** Edge of one Mantaflow cell in cm: the longest side over the resolution. */
export function gasCellCm(gas) {
  if (!gas?.scale || !Number(gas.resolution)) return null;
  return Math.round(((Math.max(...gas.scale) * 2) / Number(gas.resolution)) * 100);
}

export function formatDuration(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value)) return "";
  if (value < 90) return `${Math.max(1, Math.round(value))} 秒`;
  if (value < 5400) return `${Math.round(value / 60)} 分钟`;
  return `${Math.round(value / 360) / 10} 小时`;
}

/**
 * Grow the automatic box over every piece it misses, 0.15 m past each, and
 * keep it inside the scene. ``missed`` names pieces still outside.
 */
export function fitGasToItems(base, items, domain, maxResolution) {
  const start = boundsOfGas(base);
  if (!start || !domain) return null;
  const bounds = [...start];
  for (const item of items) {
    if (!item?.size || item.size.length !== 3) continue;
    const box = yawedBox(item);
    if (overlaps(box, start)) continue;
    bounds[0] = Math.min(bounds[0], box[0] - 0.15);
    bounds[1] = Math.max(bounds[1], box[1] + 0.15);
    bounds[2] = Math.min(bounds[2], box[2] - 0.15);
    bounds[3] = Math.max(bounds[3], box[3] + 0.15);
    bounds[4] = Math.min(bounds[4], box[4]);
    bounds[5] = Math.max(bounds[5], box[5] + 0.2);
  }
  const lo = [0, 2, 4].map((axis) => Math.max(0, bounds[axis]));
  const hi = [1, 3, 5].map((axis, index) => Math.min(domain[index], bounds[axis]));
  const gas = clampGas({
    location: [0, 1, 2].map((index) => (lo[index] + hi[index]) / 2),
    scale: [0, 1, 2].map((index) => (hi[index] - lo[index]) / 2),
    resolution: base.resolution,
  }, domain, maxResolution);
  const fitted = boundsOfGas(gas);
  const missed = items.filter((item) => item?.size?.length === 3 && !overlaps(yawedBox(item), fitted));
  return { gas, missed };
}

export function pieceIssues(item, items, shell, domainCap, gasBounds) {
  if (!item?.size || item.size.length !== 3 || !shell?.origin || !shell?.domain) return [];
  const issues = [];
  if (blocksShaft(item, shell)) issues.push("shaft");
  if (!insideDomain(item, shell.domain)) issues.push("outside");
  const outsideGas = gasBounds ? !overlaps(yawedBox(item), gasBounds) : farFromFire(item, items, shell.origin, domainCap);
  if (outsideGas) issues.push("far");
  return issues;
}

export function yawToDegrees(yaw) {
  if (!Number.isFinite(yaw)) return 0;
  return Math.round((yaw * 180) / Math.PI * 10) / 10;
}

export function degreesToYaw(value) {
  const degrees = Number(value);
  if (!Number.isFinite(degrees)) return null;
  return (degrees * Math.PI) / 180;
}

function formatMeter(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  return String(Math.round(number * 100) / 100);
}

const OFFSETS = [
  [1.7, 0.2], [0.2, 1.7], [1.7, 1.5], [2.4, -0.4],
  [2.6, 1.2], [1.2, 2.6], [-1.6, 1.4], [1.4, -1.6],
  [3.1, 0.6], [0.6, 3.1], [3.2, 2.2], [2.2, 3.2],
];

function footprint(size, x, y) {
  const hx = size[0] / 2;
  const hy = size[1] / 2;
  const hz = size[2] / 2;
  return [x - hx, x + hx, y - hy, y + hy, 0, hz * 2];
}

export function spotError(shell, items, size, x, y) {
  if (!shell?.domain || !size || size.length !== 3) return "场景还在加载";
  const [length, width, height] = shell.domain;
  const [ox, oy] = shell.origin;
  const half = shell.shaft_half_m || 1;
  const box = footprint(size, x, y);
  if (box[0] < 0.05 || box[1] > length - 0.05 || box[2] < 0.05 || box[3] > width - 0.05 || box[5] > height) return "超出场景";
  const shaft = [ox - half, ox + half, oy - half, oy + half, 0, height];
  if (overlaps(box, shaft)) return "挡住火柱";
  if (items.some((other) => overlaps(box, yawedBox(other)))) return "附近没有空位了，先挪开一件";
  return "";
}

function findSpot(shell, items, size) {
  const [ox, oy] = shell.origin;
  for (let ring = 0; ring < 4; ring += 1) {
    const scale = 1 + ring * 0.45;
    for (const [dx, dy] of OFFSETS) {
      const x = ox + dx * scale;
      const y = oy + dy * scale;
      if (!spotError(shell, items, size, x, y)) return { x, y, z: size[2] / 2 };
    }
  }
  return null;
}

function placementError(shell, items, asset, maxPlacements) {
  if (!shell) return "场景还在加载";
  if (items.length >= maxPlacements) return `最多摆 ${maxPlacements} 件`;
  if (!asset.size || asset.size.length !== 3) return "这件资产没有尺寸";
  return "";
}

function placedItem(asset, items, x, y) {
  const count = items.filter((item) => item.asset_id === asset.asset_id).length + 1;
  return {
    asset_id: asset.asset_id,
    name: `${asset.label || "燃料"}${count}`,
    x,
    y,
    z: asset.size[2] / 2,
    yaw: 0,
    size: [...asset.size],
    fuel_type: asset.fuel_type || "木材",
    ignition: false,
  };
}

export function placeAsset(shell, items, asset, maxPlacements) {
  const error = placementError(shell, items, asset, maxPlacements);
  if (error) return { error };
  const spot = findSpot(shell, items, asset.size);
  if (!spot) return { error: "附近没有空位了，先挪开一件" };
  return { item: placedItem(asset, items, spot.x, spot.y) };
}

export function placeAssetAt(shell, items, asset, maxPlacements, x, y) {
  const error = placementError(shell, items, asset, maxPlacements);
  if (error) return { error };
  const blocked = spotError(shell, items, asset.size, x, y);
  if (blocked) return { error: blocked };
  return { item: placedItem(asset, items, x, y) };
}

export function assetRecord(assets, item) {
  const found = (assets || []).find((asset) => asset.asset_id === item.asset_id);
  if (found) return found;
  if (String(item.asset_id || "").startsWith("upload:")) {
    const token = String(item.asset_id).split(":")[1];
    return { asset_id: item.asset_id, label: item.name || "上传", size: item.size, url: `/assets/uploads/${token}.glb`, fuel_type: item.fuel_type };
  }
  return { asset_id: item.asset_id, label: item.name || "燃料", size: item.size, url: "", fuel_type: item.fuel_type };
}

export function placementsWithSize(items, assets) {
  return (items || []).map((item) => {
    if (Array.isArray(item?.size) && item.size.length === 3) return item;
    const size = assetRecord(assets, item)?.size;
    if (!Array.isArray(size) || size.length !== 3) return item;
    return { ...item, size };
  });
}

// Bounding box of the object's contents in its own local frame, ignoring the
// object's position, rotation, and scale. Parent transforms cancel out, so a
// model sitting away from the world origin still centers on its group.
function contentBox(object) {
  object.updateMatrixWorld(true);
  const inverse = new THREE.Matrix4().copy(object.matrixWorld).invert();
  const box = new THREE.Box3();
  let found = false;
  object.traverse((node) => {
    const geometry = node.geometry;
    if (!node.isMesh || !geometry) return;
    if (!geometry.boundingBox) geometry.computeBoundingBox();
    const transformed = geometry.boundingBox.clone();
    transformed.applyMatrix4(new THREE.Matrix4().multiplyMatrices(inverse, node.matrixWorld));
    if (!found) {
      box.copy(transformed);
      found = true;
    } else box.union(transformed);
  });
  return found ? box : null;
}

export function fitObject(object, size) {
  object.position.set(0, 0, 0);
  object.rotation.set(0, 0, 0);
  object.scale.set(1, 1, 1);
  const box = contentBox(object);
  if (!box) return;
  const span = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const sx = size[0] / Math.max(span.x, 1e-4);
  const sy = size[2] / Math.max(span.y, 1e-4);
  const sz = size[1] / Math.max(span.z, 1e-4);
  object.scale.set(sx, sy, sz);
  object.position.set(-center.x * sx, -center.y * sy, -center.z * sz);
  object.updateMatrixWorld(true);
}

const thumbCache = new Map();
let thumbRenderer = null;
let thumbChain = Promise.resolve();
const thumbLoader = new GLTFLoader();
const FLOOR_PLANE = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);

function shadeHex(hex, scale) {
  const value = Number.parseInt(String(hex).replace("#", ""), 16);
  const channel = (shift) => Math.max(0, Math.min(255, Math.round(((value >> shift) & 255) * scale)));
  return `rgb(${channel(16)}, ${channel(8)}, ${channel(0)})`;
}

function boxThumb(fuel) {
  const key = `box:${fuel || "木材"}`;
  const cached = thumbCache.get(key);
  if (cached) return cached;
  const canvas = document.createElement("canvas");
  canvas.width = 96;
  canvas.height = 96;
  const ctx = canvas.getContext("2d");
  const color = FUEL_COLOR[fuel] || "#c4894e";
  ctx.fillStyle = "#18231f";
  ctx.fillRect(0, 0, 96, 96);
  const face = (points, fill) => {
    ctx.beginPath();
    ctx.moveTo(points[0], points[1]);
    for (let index = 2; index < points.length; index += 2) ctx.lineTo(points[index], points[index + 1]);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
  };
  face([48, 22, 76, 36, 48, 50, 20, 36], shadeHex(color, 1.18));
  face([20, 36, 48, 50, 48, 76, 20, 62], shadeHex(color, 0.72));
  face([48, 50, 76, 36, 76, 62, 48, 76], color);
  const url = canvas.toDataURL("image/png");
  thumbCache.set(key, url);
  return url;
}

function getThumbRenderer() {
  if (thumbRenderer) return thumbRenderer;
  thumbRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
  thumbRenderer.setSize(96, 96, false);
  thumbRenderer.setClearColor(0x18231f, 1);
  thumbRenderer.outputColorSpace = THREE.SRGBColorSpace;
  return thumbRenderer;
}

function renderModelThumb(object, size) {
  const root = new THREE.Group();
  root.add(object);
  fitObject(object, size);
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root);
  const center = box.getCenter(new THREE.Vector3());
  const span = box.getSize(new THREE.Vector3());
  const radius = Math.max(span.x, span.y, span.z, 0.2) * 0.72;
  const camera = new THREE.PerspectiveCamera(30, 1, Math.max(radius / 30, 0.01), radius * 24);
  camera.position.set(center.x + radius * 1.55, center.y + radius * 1.15, center.z + radius * 1.55);
  camera.lookAt(center);
  const scene = new THREE.Scene();
  scene.add(root);
  scene.add(new THREE.AmbientLight("#ffffff", 0.86));
  const sun = new THREE.DirectionalLight("#ffffff", 1.25);
  sun.position.set(radius, radius * 2.2, radius);
  scene.add(sun);
  const renderer = getThumbRenderer();
  renderer.render(scene, camera);
  const url = renderer.domElement.toDataURL("image/png");
  scene.clear();
  return url;
}

function scheduleThumb(asset) {
  const job = thumbChain.then(() => new Promise((resolve) => {
    thumbLoader.load(asset.url, (gltf) => {
      try {
        resolve(renderModelThumb(gltf.scene.clone(true), asset.size));
      } catch {
        resolve(boxThumb(asset.fuel_type));
      }
    }, undefined, () => resolve(boxThumb(asset.fuel_type)));
  }));
  thumbChain = job.then(() => {}, () => {});
  return job;
}

function useAssetThumb(asset) {
  const id = asset?.asset_id || "";
  const url = asset?.url || "";
  const fuel = asset?.fuel_type || "";
  const sizeKey = Array.isArray(asset?.size) ? asset.size.join(",") : "";
  const fallback = boxThumb(fuel);
  const [src, setSrc] = useState(() => thumbCache.get(id) || fallback);
  useEffect(() => {
    if (!id) return undefined;
    let cancel = false;
    const cached = thumbCache.get(id);
    if (cached) {
      setSrc(cached);
      return undefined;
    }
    const publish = (next) => {
      thumbCache.set(id, next);
      if (!cancel) setSrc(next);
    };
    if (!url || !sizeKey) {
      publish(boxThumb(fuel));
      return () => { cancel = true; };
    }
    const size = sizeKey.split(",").map(Number);
    scheduleThumb({ asset_id: id, url, fuel_type: fuel, size }).then(publish);
    return () => { cancel = true; };
  }, [id, url, fuel, sizeKey]);
  return src || fallback;
}

export function AssetThumb({ asset, className }) {
  const src = useAssetThumb(asset);
  return <img className={className} src={src} alt="" draggable={false} />;
}

class ModelBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { failed: false };
  }

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    if (this.state.failed) return this.props.fallback;
    return this.props.children;
  }
}

function FittedModel({ url, size }) {
  const gltf = useGLTF(url);
  const clone = useMemo(() => gltf.scene.clone(true), [gltf.scene]);
  useLayoutEffect(() => {
    fitObject(clone, size);
  }, [clone, size]);
  return <primitive object={clone} />;
}

function PackageBody({ item, asset, selected }) {
  const color = selected ? "#d65a3d" : (FUEL_COLOR[item.fuel_type] || "#c4894e");
  const box = (
    <mesh>
      <boxGeometry args={[item.size[0], item.size[2], item.size[1]]} />
      <meshStandardMaterial color={color} wireframe={Boolean(asset?.url)} transparent opacity={asset?.url ? 0.28 : 0.92} />
    </mesh>
  );
  if (!asset?.url) return box;
  return (
    <ModelBoundary fallback={box}>
      <Suspense fallback={box}>
        <FittedModel url={asset.url} size={item.size} />
        {box}
      </Suspense>
    </ModelBoundary>
  );
}

// TransformControls mirrors an arrow onto the near side when its positive axis
// points away from the camera. The head then aims back at the pivot, so the
// arrow reads backwards and dragging toward it lowers the coordinate. Keep the
// head on the positive side instead. The gizmo bakes that mirrored scale into
// matrixWorld before this runs, so each handle has to rebuild its matrix or the
// shaft stays on the near side while the head jumps back.
function unflipPositiveArrows(controls) {
  const root = controls?.gizmo;
  if (!root) return;
  const groups = [root.gizmo?.translate, root.picker?.translate, root.helper?.translate];
  for (const group of groups) {
    if (!group) continue;
    const buckets = new Map();
    for (const handle of group.children) {
      if (!buckets.has(handle.name)) buckets.set(handle.name, []);
      buckets.get(handle.name).push(handle);
    }
    for (const handles of buckets.values()) {
      const flipped = handles.some((handle) => handle.scale.x < 0 || handle.scale.y < 0 || handle.scale.z < 0);
      if (!flipped) continue;
      const hidden = handles.every((handle) => Math.max(Math.abs(handle.scale.x), Math.abs(handle.scale.y), Math.abs(handle.scale.z)) < 1e-4);
      if (hidden) continue;
      for (const handle of handles) {
        handle.scale.set(Math.abs(handle.scale.x), Math.abs(handle.scale.y), Math.abs(handle.scale.z));
        if (handle.tag === "bwd") handle.visible = false;
        if (handle.tag === "fwd") handle.visible = true;
        handle.updateMatrixWorld(true);
      }
    }
  }
}

function usePositiveArrows() {
  const controls = useRef(null);
  useLayoutEffect(() => {
    const current = controls.current;
    if (!current || current.__fpUnflip) return undefined;
    const original = current.updateMatrixWorld.bind(current);
    current.updateMatrixWorld = function updateMatrixWorld(...args) {
      original(...args);
      if (current.mode === "translate") unflipPositiveArrows(current);
    };
    current.__fpUnflip = true;
    return undefined;
  });
  return controls;
}

function ControlBridge({ object, proxy, mode, onDragChange, onCommit }) {
  const controls = usePositiveArrows();
  if (mode === "rotate") {
    const straighten = () => {
      object.rotation.x = 0;
      object.rotation.z = 0;
    };
    return (
      <TransformControls
        ref={controls}
        object={object}
        mode="rotate"
        showX={false}
        showY
        showZ={false}
        space="world"
        onMouseDown={() => onDragChange(true)}
        onObjectChange={straighten}
        onMouseUp={() => {
          onDragChange(false);
          straighten();
          onCommit({ yaw: object.rotation.y });
        }}
      />
    );
  }
  // proxy lives under Rx(-π/2) with no yaw, so local X/Y/Z are scene +X, +Y, and up.
  // Follow the mesh from proxy.position. Do not write proxy.position here: the
  // gizmo applies the whole drag from the pointer-down position on the next move.
  const follow = () => {
    const [x, y, z] = sceneToThree(proxy.position.x, proxy.position.y, proxy.position.z);
    object.position.set(x, y, z);
    object.rotation.x = 0;
    object.rotation.z = 0;
  };
  return (
    <TransformControls
      ref={controls}
      object={proxy}
      mode="translate"
      showX
      showY
      showZ
      space="local"
      onMouseDown={() => onDragChange(true)}
      onObjectChange={follow}
      onMouseUp={() => {
        onDragChange(false);
        follow();
        onCommit({ x: proxy.position.x, y: proxy.position.y, z: proxy.position.z, yaw: object.rotation.y });
      }}
    />
  );
}

function Package({ item, asset, selected, mode, issues, onSelect, onCommit }) {
  const group = useRef(null);
  const basis = useRef(null);
  const proxy = useRef(null);
  const dragging = useRef(false);
  const [ready, setReady] = useState(false);
  useLayoutEffect(() => { setReady(true); }, []);
  useLayoutEffect(() => {
    const object = group.current;
    if (!object || dragging.current) return;
    const [x, y, z] = sceneToThree(item.x, item.y, item.z);
    object.position.set(x, y, z);
    object.rotation.set(0, item.yaw || 0, 0);
  }, [item.x, item.y, item.z, item.yaw]);
  useLayoutEffect(() => {
    const handle = proxy.current;
    const root = basis.current;
    if (!handle || !root || dragging.current) return;
    handle.position.set(item.x, item.y, item.z);
    handle.rotation.set(0, 0, 0);
    root.updateWorldMatrix(true, true);
  }, [item.x, item.y, item.z]);
  if (!item?.size || item.size.length !== 3) return null;
  const flag = issues.includes("shaft") ? "shaft" : issues.includes("outside") ? "outside" : issues.includes("far") ? "far" : "";
  return (
    <>
      <group ref={group} userData={{ pick: onSelect }} onClick={(event) => selectClick(event, onSelect)}>
        <PackageBody item={item} asset={asset} selected={selected} />
        {flag && <SceneFlag position={[0, item.size[2] / 2 + 0.2, 0]} label={ISSUE_LABEL[flag]} />}
      </group>
      {/* Local (x, y, z) is scene (x, y, z). Rx(-π/2) puts local +Y on Three -Z and local +Z on up. */}
      <group ref={basis} rotation={[-Math.PI / 2, 0, 0]}>
        <group ref={proxy} />
      </group>
      {selected && ready && group.current && proxy.current && (
        <ControlBridge
          object={group.current}
          proxy={proxy.current}
          mode={mode}
          onDragChange={(value) => { dragging.current = value; }}
          onCommit={onCommit}
        />
      )}
    </>
  );
}

function selectClick(event, select) {
  event.stopPropagation();
  // A drag across a wall or the floor is not a selection. r3f still emits
  // click when the pointer went down on that mesh.
  if (event.delta > 2) return;
  select();
}

function ShellBox({ obstacle, xb, selected, onSelect }) {
  const size = [xb[1] - xb[0], xb[5] - xb[4], xb[3] - xb[2]];
  const center = [(xb[0] + xb[1]) / 2, (xb[2] + xb[3]) / 2, (xb[4] + xb[5]) / 2];
  const color = selected ? "#d65a3d" : obstacle.kind === "ignition" ? "#d65a3d" : obstacle.kind === "fuel" ? "#a7844a" : "#9aa39c";
  const opacity = selected ? 0.38 : obstacle.kind === "inert" ? 0.14 : 0.45;
  return (
    <mesh position={sceneToThree(center[0], center[1], center[2])} onClick={(event) => selectClick(event, onSelect)}>
      <boxGeometry args={size} />
      <meshStandardMaterial color={color} transparent opacity={opacity} side={THREE.DoubleSide} depthWrite={false} />
    </mesh>
  );
}

function FloorRing({ x, y, radius }) {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={sceneToThree(x, y, 0.05)}>
      <ringGeometry args={[Math.max(radius - 0.08, 0.02), radius, 80]} />
      <meshBasicMaterial color="#e07a55" transparent opacity={0.4} depthWrite={false} side={THREE.DoubleSide} />
    </mesh>
  );
}

export function cameraFor(shell, view) {
  const [length, width, height] = shell.domain;
  const target = sceneToThree(length / 2, width / 2, Math.min(height * 0.15, 2));
  if (view === "top") {
    return { position: sceneToThree(length / 2, width / 2 + 0.01, height * 1.35), target, fov: 42 };
  }
  if (view === "side") {
    return { position: sceneToThree(length / 2, -Math.max(width * 0.45, 3), Math.max(height * 0.35, 1.4)), target, fov: 42 };
  }
  return { position: sceneToThree(length / 2, -Math.max(width * 0.55, 3), Math.max(height * 0.9, 2)), target, fov: 42 };
}

export function bakeShot(shell, pose = shell?.camera) {
  if (!pose?.location || !pose?.target) return null;
  return {
    position: sceneToThree(pose.location[0], pose.location[1], pose.location[2]),
    target: sceneToThree(pose.target[0], pose.target[1], pose.target[2]),
  };
}

export function frameShot(shell, item) {
  if (item?.size?.length === 3) {
    const target = sceneToThree(item.x, item.y, item.z);
    const radius = Math.max(item.size[0], item.size[1], item.size[2], 0.4);
    const dist = radius * 2.6;
    return {
      position: [target[0] + dist * 0.9, target[1] + dist * 0.55, target[2] + dist * 0.9],
      target,
      fov: 42,
    };
  }
  return cameraFor(shell, "perspective");
}

export function focalFov(focalMm) {
  const focal = Number(focalMm);
  if (!Number.isFinite(focal) || focal <= 0) return 42;
  return (2 * Math.atan(18 / focal) * 180) / Math.PI;
}

function ViewBridge({ aim }) {
  const camera = useThree((state) => state.camera);
  const controls = useThree((state) => state.controls);
  const invalidate = useThree((state) => state.invalidate);
  useEffect(() => {
    if (!aim?.position || !aim?.target || !controls?.target) return;
    camera.position.set(aim.position[0], aim.position[1], aim.position[2]);
    controls.target.set(aim.target[0], aim.target[1], aim.target[2]);
    if (Number.isFinite(aim.fov)) {
      camera.fov = aim.fov;
      camera.updateProjectionMatrix();
    }
    controls.update();
    invalidate();
  }, [aim, camera, controls, invalidate]);
  return null;
}

function pointOnFloor(event, element, camera) {
  const rect = element.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  const ndc = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1,
  );
  const ray = new THREE.Raycaster();
  ray.setFromCamera(ndc, camera);
  const hit = new THREE.Vector3();
  if (!ray.ray.intersectPlane(FLOOR_PLANE, hit)) return null;
  const [x, y] = threeToScene(hit.x, hit.y, hit.z);
  return { x, y };
}

function DropBridge({ onHover, onDrop }) {
  const camera = useThree((state) => state.camera);
  const gl = useThree((state) => state.gl);
  const hoverRef = useRef(onHover);
  const dropRef = useRef(onDrop);
  hoverRef.current = onHover;
  dropRef.current = onDrop;
  useEffect(() => {
    const element = gl.domElement;
    const accepts = (transfer) => {
      const types = Array.from(transfer?.types || []);
      return types.includes("application/x-firephantom-asset") || types.includes("Files");
    };
    const over = (event) => {
      if (!accepts(event.dataTransfer)) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
      hoverRef.current(pointOnFloor(event, element, camera), event.dataTransfer);
    };
    const drop = (event) => {
      if (!accepts(event.dataTransfer)) return;
      event.preventDefault();
      dropRef.current(pointOnFloor(event, element, camera), event.dataTransfer);
    };
    element.addEventListener("dragover", over);
    element.addEventListener("drop", drop);
    return () => {
      element.removeEventListener("dragover", over);
      element.removeEventListener("drop", drop);
    };
  }, [camera, gl]);
  return null;
}

function CameraPyramid({ fov }) {
  const geometry = useMemo(() => {
    const depth = 1.35;
    const halfH = Math.tan((fov * Math.PI) / 360) * depth;
    const halfW = halfH * 1.2;
    const z0 = -0.55;
    const z1 = z0 - depth;
    const corners = [
      [-halfW, -halfH, z1],
      [halfW, -halfH, z1],
      [halfW, halfH, z1],
      [-halfW, halfH, z1],
    ];
    const apex = [0, 0, z0];
    const pairs = [];
    corners.forEach((corner) => pairs.push(apex, corner));
    for (let index = 0; index < corners.length; index += 1) pairs.push(corners[index], corners[(index + 1) % corners.length]);
    const positions = new Float32Array(pairs.flat());
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return geo;
  }, [fov]);
  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color="#f2d48a" />
    </lineSegments>
  );
}

function AxisGizmo({ active, position, onDrag, onCommit }) {
  const basis = useRef(null);
  const proxy = useRef(null);
  const dragging = useRef(false);
  const controls = usePositiveArrows();
  const [ready, setReady] = useState(false);
  useLayoutEffect(() => { setReady(true); }, []);
  useLayoutEffect(() => {
    const handle = proxy.current;
    const root = basis.current;
    if (!handle || !root || dragging.current || !position) return;
    handle.position.set(position[0], position[1], position[2]);
    handle.rotation.set(0, 0, 0);
    root.updateWorldMatrix(true, true);
  }, [position]);
  const read = () => {
    const handle = proxy.current;
    return handle ? [handle.position.x, handle.position.y, handle.position.z] : null;
  };
  return (
    <>
      <group ref={basis} rotation={[-Math.PI / 2, 0, 0]}>
        <group ref={proxy} />
      </group>
      {active && ready && proxy.current && (
        <TransformControls
          ref={controls}
          object={proxy.current}
          mode="translate"
          space="local"
          onMouseDown={() => { dragging.current = true; }}
          onObjectChange={() => { const point = read(); if (point) onDrag?.(point); }}
          onMouseUp={() => {
            dragging.current = false;
            const point = read();
            if (point) onCommit(point);
          }}
        />
      )}
    </>
  );
}

function BakeCamera({ pose, domain, label, selected, part, tool, onSelect, onLook, onCommit }) {
  const body = useRef(null);
  const marker = useRef(null);
  const dragging = useRef(false);
  const position = useMemo(() => sceneToThree(pose.location[0], pose.location[1], pose.location[2]), [pose]);
  const target = useMemo(() => sceneToThree(pose.target[0], pose.target[1], pose.target[2]), [pose]);
  const fov = focalFov(pose.focal_mm);
  const span = Math.max(domain?.[0] || 0, domain?.[1] || 0, 1);
  const bodyScale = Math.max(0.42, span * 0.016);
  const look = (object, at) => {
    if (!object) return;
    object.lookAt(at[0], at[1], at[2]);
    object.rotateY(Math.PI);
  };
  useLayoutEffect(() => {
    const object = body.current;
    if (!object || dragging.current) return;
    object.position.set(position[0], position[1], position[2]);
    look(object, target);
  }, [position, target]);
  useLayoutEffect(() => {
    if (!marker.current || dragging.current) return;
    marker.current.position.set(target[0], target[1], target[2]);
  }, [target]);
  const showGizmo = selected && tool === "translate";
  const gizmoPoint = part === "target" ? pose.target : pose.location;
  const follow = (point) => {
    dragging.current = true;
    const next = sceneToThree(point[0], point[1], point[2]);
    if (part === "target") {
      marker.current?.position.set(next[0], next[1], next[2]);
      look(body.current, next);
      return;
    }
    if (body.current) body.current.position.set(next[0], next[1], next[2]);
    look(body.current, target);
  };
  return (
    <group>
      <group
        ref={body}
        position={position}
        scale={bodyScale}
        userData={{ pick: () => onSelect("body"), look: onLook }}
        onClick={(event) => selectClick(event, () => onSelect("body"))}
        onDoubleClick={(event) => { event.stopPropagation(); onLook(); }}
      >
        <mesh>
          <boxGeometry args={[0.72, 0.5, 0.9]} />
          <meshStandardMaterial color={selected ? "#d65a3d" : "#e7eee9"} />
        </mesh>
        <mesh position={[0, 0, -0.62]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.18, 0.28, 0.38, 16]} />
          <meshStandardMaterial color="#314038" />
        </mesh>
        <CameraPyramid fov={fov} />
      </group>
      {selected && (
        <mesh
          ref={marker}
          position={target}
          userData={{ pick: () => onSelect("target") }}
          onClick={(event) => selectClick(event, () => onSelect("target"))}
        >
          <sphereGeometry args={[Math.max(0.08, bodyScale * 0.22), 16, 12]} />
          <meshStandardMaterial color={part === "target" ? "#d65a3d" : "#f2d48a"} emissive="#f2d48a" emissiveIntensity={0.35} />
        </mesh>
      )}
      <SceneFlag position={[position[0], position[1] + bodyScale * 0.85, position[2]]} label={label} picked={selected} />
      <AxisGizmo
        active={showGizmo}
        position={gizmoPoint}
        onDrag={follow}
        onCommit={(point) => {
          dragging.current = false;
          const accepted = onCommit(part === "target" ? { target: point } : { location: point });
          if (accepted === false) {
            if (part === "target") marker.current?.position.set(target[0], target[1], target[2]);
            else if (body.current) {
              body.current.position.set(position[0], position[1], position[2]);
              look(body.current, target);
            }
          }
        }}
      />
    </group>
  );
}

function pickData(object) {
  let node = object;
  while (node) {
    if (node.userData?.pick || node.userData?.look) return node.userData;
    node = node.parent;
  }
  return null;
}

function relayPick(event, key, fallback) {
  const data = event.intersections.map((hit) => pickData(hit.object)).find((item) => item?.[key]);
  event.stopPropagation();
  if (data?.[key]) data[key]();
  else if (fallback) fallback();
}

function GasDomain({ gas, selected, tool, onSelect, onCommit }) {
  const basis = useRef(null);
  const proxy = useRef(null);
  const mesh = useRef(null);
  const dragging = useRef(false);
  const controls = usePositiveArrows();
  const [ready, setReady] = useState(false);
  const edges = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), []);
  const lines = useRef(null);
  useLayoutEffect(() => {
    const line = lines.current;
    if (!line) return undefined;
    const base = THREE.Line.prototype.raycast;
    line.raycast = (raycaster, intersects) => {
      const params = raycaster.params.Line;
      const previous = params.threshold;
      params.threshold = 0.15;
      base.call(line, raycaster, intersects);
      params.threshold = previous;
    };
    return () => { line.raycast = base; };
  }, []);
  useLayoutEffect(() => { setReady(true); }, []);
  useLayoutEffect(() => {
    const handle = proxy.current;
    const root = basis.current;
    const object = mesh.current;
    if (!handle || !root || !object || dragging.current || !gas) return;
    handle.position.set(gas.location[0], gas.location[1], gas.location[2]);
    handle.scale.set(gas.scale[0] * 2, gas.scale[1] * 2, gas.scale[2] * 2);
    handle.rotation.set(0, 0, 0);
    root.updateWorldMatrix(true, true);
    object.position.copy(handle.position);
    object.scale.copy(handle.scale);
    object.rotation.set(0, 0, 0);
  }, [gas]);
  const show = selected && (tool === "translate" || tool === "scale");
  const follow = () => {
    const handle = proxy.current;
    const object = mesh.current;
    if (!handle || !object) return;
    object.position.copy(handle.position);
    object.scale.copy(handle.scale);
  };
  const color = selected ? "#d65a3d" : "#e07a55";
  const sizeLabel = gas.scale.map((edge) => formatMeter(edge * 2)).join(" × ") + " m";
  // The handle lives outside the rotated group. three-stdlib writes the
  // object's world position into the gizmo's local position, so a parent
  // rotation would move the arrows off the box.
  return (
    <>
      <group ref={basis} rotation={[-Math.PI / 2, 0, 0]}>
        <group ref={proxy} />
        <mesh ref={mesh} onClick={(event) => relayPick(event, "pick", onSelect)} onDoubleClick={(event) => relayPick(event, "look")}>
          <boxGeometry args={[1, 1, 1]} />
          <meshBasicMaterial color={color} transparent opacity={0.12} depthWrite={false} />
          <lineSegments ref={lines} geometry={edges}>
            <lineBasicMaterial color={color} />
          </lineSegments>
        </mesh>
      </group>
      {selected && <SceneFlag position={sceneToThree(gas.location[0], gas.location[1], gas.location[2] + gas.scale[2] + 0.15)} label={sizeLabel} picked />}
      {show && ready && proxy.current && (
        <TransformControls
          ref={controls}
          object={proxy.current}
          mode={tool === "scale" ? "scale" : "translate"}
          space="local"
          onMouseDown={() => { dragging.current = true; }}
          onObjectChange={follow}
          onMouseUp={() => {
            dragging.current = false;
            const handle = proxy.current;
            if (!handle) return;
            onCommit({
              location: [handle.position.x, handle.position.y, handle.position.z],
              scale: [handle.scale.x / 2, handle.scale.y / 2, handle.scale.z / 2],
            });
          }}
        />
      )}
    </>
  );
}

function StructureHandle({ center, size, role, label, tool, onCommit }) {
  const basis = useRef(null);
  const proxy = useRef(null);
  const mesh = useRef(null);
  const dragging = useRef(false);
  const controls = usePositiveArrows();
  const [ready, setReady] = useState(false);
  const edges = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), []);
  useLayoutEffect(() => { setReady(true); }, []);
  useLayoutEffect(() => {
    const handle = proxy.current;
    const root = basis.current;
    const object = mesh.current;
    if (!handle || !root || !object || dragging.current || !center || !size) return;
    // Local position is scene (x, y, z). Rx(-π/2) maps it to Three (x, z, -y),
    // the same conversion as sceneToThree. Scale is the full edge.
    handle.position.set(center[0], center[1], center[2]);
    handle.scale.set(size[0], size[1], size[2]);
    handle.rotation.set(0, 0, 0);
    root.updateWorldMatrix(true, true);
    object.position.copy(handle.position);
    object.scale.copy(handle.scale);
    object.rotation.set(0, 0, 0);
  }, [center, size]);
  const show = tool === "translate" || tool === "scale";
  const follow = () => {
    const handle = proxy.current;
    const object = mesh.current;
    if (!handle || !object) return;
    object.position.copy(handle.position);
    object.scale.copy(handle.scale);
  };
  const color = role === "ignition" ? "#d65a3d" : role === "fuel" ? "#a7844a" : "#d7c4a4";
  // The gizmo stays outside the rotated group. three-stdlib writes the object's
  // world position into the gizmo's local position, so a parent rotation would
  // drop the arrows under the floor.
  return (
    <>
      <group ref={basis} rotation={[-Math.PI / 2, 0, 0]}>
        <group ref={proxy} />
        <mesh ref={mesh} onClick={(event) => selectClick(event, () => {})}>
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial color={color} transparent opacity={0.38} side={THREE.DoubleSide} depthWrite={false} />
          <lineSegments geometry={edges}>
            <lineBasicMaterial color="#d65a3d" />
          </lineSegments>
        </mesh>
      </group>
      <SceneFlag position={sceneToThree(center[0], center[1], center[2] + size[2] / 2 + 0.15)} label={label} picked />
      {show && ready && proxy.current && (
        <TransformControls
          ref={controls}
          object={proxy.current}
          mode={tool === "scale" ? "scale" : "translate"}
          space="local"
          onMouseDown={() => { dragging.current = true; }}
          onObjectChange={follow}
          onMouseUp={() => {
            dragging.current = false;
            const handle = proxy.current;
            if (!handle) return;
            onCommit({
              center: [handle.position.x, handle.position.y, handle.position.z],
              size: [Math.abs(handle.scale.x), Math.abs(handle.scale.y), Math.abs(handle.scale.z)],
            });
          }}
        />
      )}
    </>
  );
}

function OriginMarker({ origin, selected, tool, onSelect, onCommit }) {
  const mesh = useRef(null);
  const dragging = useRef(false);
  const position = sceneToThree(origin[0], origin[1], origin[2]);
  useLayoutEffect(() => {
    if (!mesh.current || dragging.current) return;
    mesh.current.position.set(position[0], position[1], position[2]);
  }, [position]);
  return (
    <group>
      <mesh ref={mesh} position={position} userData={{ pick: onSelect }} onClick={(event) => selectClick(event, onSelect)}>
        <octahedronGeometry args={[0.16, 0]} />
        <meshStandardMaterial color={selected ? "#ffb199" : "#d65a3d"} emissive="#d65a3d" emissiveIntensity={0.7} />
      </mesh>
      <SceneFlag position={sceneToThree(origin[0], origin[1], origin[2] + 0.35)} label="起火点" picked={selected} />
      <AxisGizmo
        active={selected && tool === "translate"}
        position={origin}
        onDrag={(point) => {
          dragging.current = true;
          const next = sceneToThree(point[0], point[1], point[2]);
          mesh.current?.position.set(next[0], next[1], next[2]);
        }}
        onCommit={(point) => {
          dragging.current = false;
          onCommit(point);
        }}
      />
    </group>
  );
}

function GhostModel({ url, size }) {
  const gltf = useGLTF(url);
  const clone = useMemo(() => {
    const next = gltf.scene.clone(true);
    next.traverse((node) => {
      if (!node.isMesh || !node.material) return;
      const material = node.material.clone();
      material.transparent = true;
      material.opacity = 0.45;
      material.depthWrite = false;
      node.material = material;
    });
    return next;
  }, [gltf.scene]);
  useLayoutEffect(() => { fitObject(clone, size); }, [clone, size]);
  return <primitive object={clone} />;
}

function PlacementGhost({ ghost }) {
  if (!ghost) return null;
  const size = ghost.asset?.size || [0.8, 0.8, 0.8];
  const position = sceneToThree(ghost.x, ghost.y, size[2] / 2);
  const color = ghost.error ? "#b44b32" : (FUEL_COLOR[ghost.asset?.fuel_type] || "#c4894e");
  return (
    <group position={position}>
      <mesh>
        <boxGeometry args={[size[0], size[2], size[1]]} />
        <meshStandardMaterial color={color} transparent opacity={0.35} depthWrite={false} />
      </mesh>
      {ghost.asset?.url && (
        <Suspense fallback={null}>
          <GhostModel url={ghost.asset.url} size={size} />
        </Suspense>
      )}
    </group>
  );
}

function CanvasBridge({ onCanvas }) {
  const gl = useThree((state) => state.gl);
  useEffect(() => {
    if (!onCanvas) return undefined;
    onCanvas(gl.domElement);
    return () => onCanvas(null);
  }, [gl, onCanvas]);
  return null;
}

function sceneEulerToThreeQuaternion(rotation) {
  const euler = new THREE.Euler(rotation?.[0] || 0, rotation?.[1] || 0, rotation?.[2] || 0, "XYZ");
  const scene = new THREE.Matrix4().makeRotationFromEuler(euler);
  const column = (index) => {
    const vector = new THREE.Vector3().setFromMatrixColumn(scene, index);
    return new THREE.Vector3(vector.x, vector.z, -vector.y);
  };
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(column(0), column(1), column(2)));
}

function threeQuaternionToSceneEuler(quaternion) {
  const matrix = new THREE.Matrix4().makeRotationFromQuaternion(quaternion);
  const column = (index) => {
    const vector = new THREE.Vector3().setFromMatrixColumn(matrix, index);
    return new THREE.Vector3(vector.x, -vector.z, vector.y);
  };
  const scene = new THREE.Matrix4().makeBasis(column(0), column(1), column(2));
  const euler = new THREE.Euler().setFromRotationMatrix(scene, "XYZ");
  return [round4(euler.x), round4(euler.y), round4(euler.z)];
}

function rgbCss(color) {
  const channel = (value) => Math.round(Math.min(1, Math.max(0, Number(value) || 0)) * 255);
  return `rgb(${channel(color?.[0])}, ${channel(color?.[1])}, ${channel(color?.[2])})`;
}

export function rgbToHex(color) {
  const channel = (value) => Math.round(Math.min(1, Math.max(0, Number(value) || 0)) * 255).toString(16).padStart(2, "0");
  return `#${channel(color?.[0])}${channel(color?.[1])}${channel(color?.[2])}`;
}

export function hexToRgb(hex) {
  const text = String(hex || "").replace("#", "");
  if (text.length !== 6) return [1, 1, 1];
  return [0, 2, 4].map((start) => round4(Number.parseInt(text.slice(start, start + 2), 16) / 255));
}

function lightSignature(light) {
  return JSON.stringify({
    id: light.id,
    name: light.name || "",
    type: light.type,
    color: (light.color || []).map(round4),
    energy: round4(light.energy),
    size: round4(light.size),
    size_y: round4(light.size_y),
    spot_size: round4(light.spot_size),
    spot_blend: round4(light.spot_blend),
    location: (light.location || []).map(round4),
    rotation: (light.rotation || [0, 0, 0]).map(round4),
    enabled: Boolean(light.enabled),
  });
}

export function sameLights(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((light, index) => lightSignature(light) === lightSignature(right[index]));
}

export function clampLightPoint(point, domain) {
  return point.map((value, index) => {
    const limit = Number(domain?.[index]);
    if (!Number.isFinite(limit)) return round4(value);
    if (limit <= 0.1) return round4(limit / 2);
    return round4(Math.min(Math.max(Number(value), 0.05), limit - 0.05));
  });
}

function SunLamp({ color, intensity }) {
  const lamp = useRef(null);
  const target = useRef(null);
  useLayoutEffect(() => {
    if (lamp.current && target.current) lamp.current.target = target.current;
  });
  return (
    <>
      <directionalLight ref={lamp} color={color} intensity={intensity} />
      <object3D ref={target} position={[0, 0, -1]} />
    </>
  );
}

// Blender's glTF exporter in its compat lighting mode: a sun's strength is
// already per square meter, and point and spot watts spread over 4π sr. An
// area lamp's watts leave one face of a Lambertian rectangle, so its
// radiance is W / (π · area).
export function viewportIntensity(light) {
  if (!light?.enabled) return 0;
  const energy = Math.max(0, Number(light.energy) || 0);
  if (light.type === "SUN") return energy;
  if (light.type === "AREA") {
    const area = Math.max(0.05, Number(light.size) || 0.5) * Math.max(0.05, Number(light.size_y) || 0.5);
    return energy / (Math.PI * area);
  }
  return energy / (4 * Math.PI);
}

function SceneLight({ light, lit, shown = true, selected, tool, onSelect, onCommit }) {
  const marker = useRef(null);
  const dragging = useRef(false);
  const controls = usePositiveArrows();
  const [ready, setReady] = useState(false);
  const position = sceneToThree(light.location[0], light.location[1], light.location[2]);
  const [rx, ry, rz] = light.rotation || [0, 0, 0];
  const quaternion = useMemo(() => sceneEulerToThreeQuaternion([rx, ry, rz]), [rx, ry, rz]);
  const directed = light.type !== "POINT";
  const color = rgbCss(light.color || [1, 1, 1]);
  // Lamps stay mounted at zero when off or hidden. Adding or removing one
  // changes the light count, and three then recompiles every material.
  const viewportEnergy = lit ? viewportIntensity(light) : 0;
  useLayoutEffect(() => { setReady(true); }, []);
  useLayoutEffect(() => {
    const object = marker.current;
    if (!object || dragging.current) return;
    object.position.set(position[0], position[1], position[2]);
    object.quaternion.copy(quaternion);
  }, [position, quaternion]);
  // Hiding drops the marker and gizmos. The lamp itself stays, see above.
  const showMove = shown && selected && tool !== "rotate" && tool !== "scale";
  const showTurn = shown && selected && tool === "rotate" && directed;
  return (
    <>
      <group ref={marker}>
        {light.type === "POINT" && <pointLight color={color} intensity={viewportEnergy} decay={2} />}
        {light.type === "SUN" && <SunLamp color={color} intensity={viewportEnergy} />}
        {light.type === "SPOT" && (
          <spotLight
            color={color}
            intensity={viewportEnergy}
            angle={Math.min(Math.PI / 2 - 0.01, (Number(light.spot_size) || 0.8) / 2)}
            penumbra={Number(light.spot_blend) || 0}
            decay={2}
          />
        )}
        {light.type === "AREA" && (
          <rectAreaLight
            color={color}
            intensity={viewportEnergy}
            width={Math.max(0.05, Number(light.size) || 0.5)}
            height={Math.max(0.05, Number(light.size_y) || 0.5)}
          />
        )}
        {shown && (
          <mesh rotation={light.type === "SPOT" ? [-Math.PI / 2, 0, 0] : [0, 0, 0]} onClick={(event) => selectClick(event, onSelect)}>
            {light.type === "AREA" ? <planeGeometry args={[0.28, 0.18]} /> : light.type === "SPOT" ? <coneGeometry args={[0.07, 0.18, 12]} /> : <sphereGeometry args={[0.08, 16, 12]} />}
            <meshStandardMaterial color={selected ? "#d65a3d" : color} emissive={color} emissiveIntensity={light.enabled ? 0.55 : 0.05} />
          </mesh>
        )}
        {shown && directed && light.type !== "SPOT" && (
          <mesh position={[0, 0, -0.18]}>
            <sphereGeometry args={[0.03, 10, 8]} />
            <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.4} />
          </mesh>
        )}
        {shown && <SceneFlag position={[0, 0.22, 0]} label={light.name || LIGHT_LABEL[light.type]} picked={selected} />}
      </group>
      <AxisGizmo
        active={showMove}
        position={light.location}
        onDrag={(point) => {
          dragging.current = true;
          const next = sceneToThree(point[0], point[1], point[2]);
          marker.current?.position.set(next[0], next[1], next[2]);
        }}
        onCommit={(point) => {
          dragging.current = false;
          onCommit({ location: point.map(round4) });
        }}
      />
      {showTurn && ready && marker.current && (
        <TransformControls
          ref={controls}
          object={marker.current}
          mode="rotate"
          space="local"
          onMouseDown={() => { dragging.current = true; }}
          onMouseUp={() => {
            dragging.current = false;
            const object = marker.current;
            if (!object) return;
            onCommit({ rotation: threeQuaternionToSceneEuler(object.quaternion) });
          }}
        />
      )}
    </>
  );
}

export const SHADING = [["solid", "实体"], ["material", "材质预览"], ["rendered", "渲染预览"]];

const keyOffset = new THREE.Vector3();
const keyAhead = new THREE.Vector3();

/**
 * Viewport light, after Blender's shading modes. Solid is studio light that
 * follows the view and ignores the scene. Material preview adds a room
 * environment and can switch scene lamps on. Rendered is the bake's own
 * world and lamps, so a room is as dark here as in the render. Every light
 * stays mounted and only changes intensity, so switching does not recompile.
 */
function ViewportShading({ mode, world }) {
  const scene = useThree((state) => state.scene);
  const gl = useThree((state) => state.gl);
  const camera = useThree((state) => state.camera);
  const invalidate = useThree((state) => state.invalidate);
  const key = useRef(null);
  const target = useRef(null);
  useLayoutEffect(() => {
    if (key.current && target.current) key.current.target = target.current;
  }, []);
  useEffect(() => {
    if (mode !== "material") return undefined;
    const pmrem = new THREE.PMREMGenerator(gl);
    const room = new RoomEnvironment();
    const texture = pmrem.fromScene(room, 0.04).texture;
    room.dispose();
    scene.environment = texture;
    invalidate();
    return () => {
      if (scene.environment === texture) scene.environment = null;
      texture.dispose();
      pmrem.dispose();
      invalidate();
    };
  }, [mode, gl, scene, invalidate]);
  // Upper left of the view, aimed where the camera looks.
  useFrame(() => {
    const lamp = key.current;
    const aim = target.current;
    if (!lamp || !aim) return;
    keyOffset.set(-2, 3, 1);
    camera.localToWorld(keyOffset);
    lamp.position.copy(keyOffset);
    keyAhead.set(0, 0, -6);
    camera.localToWorld(keyAhead);
    aim.position.copy(keyAhead);
    aim.updateMatrixWorld();
  });
  const solid = mode === "solid";
  // A uniform world of radiance L lights a surface with irradiance πL.
  const worldColor = world?.color || [0, 0, 0];
  const peak = Math.max(...worldColor, 1e-6);
  const worldTint = rgbCss(worldColor.map((value) => value / peak));
  const worldIntensity = mode === "rendered" ? peak * (Number(world?.strength) || 0) * Math.PI : 0;
  return (
    <>
      <hemisphereLight args={["#e8eef5", "#5a5246"]} intensity={solid ? 1.5 : 0} />
      <directionalLight ref={key} intensity={solid ? 1.6 : 0} />
      <object3D ref={target} />
      <ambientLight color={worldTint} intensity={worldIntensity} />
    </>
  );
}

// Children of the Canvas re-render with the viewport. Components here move
// objects in layout effects, which r3f does not see, so ask for one frame.
function FrameOnRender() {
  const invalidate = useThree((state) => state.invalidate);
  useEffect(() => { invalidate(); });
  return null;
}

function SceneViewport({ scenario, shell, items, assets, selected, tool, aim, ghost, cameras, gas, origin, lights, shading, sceneLights, world, hidden = {}, editingId, onHover, onDrop, onLook, onSelect, onCommit, onCommitOrigin, onCommitCamera, onCommitGas, onCommitLight, onCommitStructure, onCanvas }) {
  const [length, width, height] = shell?.domain || [];
  // Key on the numbers. A new shot hands OrbitControls a new target, which
  // snaps the view back every time the page re-renders.
  const shot = useMemo(() => (length ? cameraFor({ domain: [length, width, height] }, "perspective") : null), [length, width, height]);
  const gasBounds = gas ? boundsOfGas(gas) : null;
  const issues = useMemo(
    () => items.map((item) => (shell?.origin ? pieceIssues(item, items, shell, 4, gasBounds) : [])),
    // gasBounds and shell are rebuilt each render; their contents are the key.
    [items, shell?.origin?.[0], shell?.origin?.[1], length, width, height, gasBounds?.join(",")],
  );
  const lampsLit = shading === "rendered" || (shading === "material" && sceneLights);
  // The bake cuts the flame shaft out of every solid; show the same pieces.
  const shellPieces = useMemo(() => {
    if (!shell?.origin || !length) return [];
    const shaft = shaftBox(shell);
    return (shell.obstacles || []).map((obstacle) => ({ obstacle, pieces: subtractBox(obstacle.xb, shaft) }));
  }, [shell?.obstacles, shell?.origin?.[0], shell?.origin?.[1], shell?.shaft_half_m, length, width, height]);
  const editing = (shell?.obstacles || []).find((item) => item.name === editingId) || null;
  const press = useRef(null);
  useEffect(() => {
    const down = (event) => { press.current = { x: event.clientX, y: event.clientY }; };
    window.addEventListener("pointerdown", down, true);
    return () => window.removeEventListener("pointerdown", down, true);
  }, []);
  if (!shell?.domain || !origin || !shot) {
    return <div className="empty-result"><strong>场景数据还在加载</strong><span>目录返回后可以摆放燃料。</span></div>;
  }
  return (
    <div className="preview-stage scene-viewport">
      <Canvas
        camera={{ position: shot.position, fov: 42, near: 0.05, far: 800 }}
        frameloop="demand"
        dpr={[1, 1.5]}
        gl={{ preserveDrawingBuffer: true }}
        onPointerMissed={(event) => {
          // A gizmo drag is not an r3f hit, so the click that ends it looks like
          // empty space. Keep the selection when the pointer actually moved.
          const start = press.current;
          if (start && Math.hypot(event.clientX - start.x, event.clientY - start.y) > 4) return;
          onSelect(null);
        }}
      >
        <color attach="background" args={["#18231f"]} />
        <ViewportShading mode={shading} world={world} />
        <FrameOnRender />
        {!hidden["shell:floor"] && (
          <mesh
            rotation={[-Math.PI / 2, 0, 0]}
            position={sceneToThree(length / 2, width / 2, 0.01)}
            onClick={(event) => selectClick(event, () => onSelect({ kind: "shell", id: "floor" }))}
          >
            <planeGeometry args={[length, width]} />
            <meshStandardMaterial color={selected?.kind === "shell" && selected.id === "floor" ? "#d7c4a4" : (FLOOR[scenario] || "#6e7472")} />
          </mesh>
        )}
        {shellPieces.filter(({ obstacle }) => obstacle.name !== editingId && shellDrawn(obstacle, hidden, selected)).flatMap(({ obstacle, pieces }) => pieces.map((xb, index) => (
          <ShellBox
            key={`${obstacle.name}-${index}`}
            obstacle={obstacle}
            xb={xb}
            selected={selected?.kind === "shell" && selected.id === obstacle.name}
            onSelect={() => onSelect({ kind: "shell", id: obstacle.name })}
          />
        )))}
        {editing?.xb && (
          <StructureHandle
            center={editing.center || [(editing.xb[0] + editing.xb[1]) / 2, (editing.xb[2] + editing.xb[3]) / 2, (editing.xb[4] + editing.xb[5]) / 2]}
            size={editing.size || [editing.xb[1] - editing.xb[0], editing.xb[3] - editing.xb[2], editing.xb[5] - editing.xb[4]]}
            role={editing.kind}
            label={editing.label || SHELL_LABEL[editing.name] || editing.name}
            tool={tool}
            onCommit={(patch) => onCommitStructure(editing.name, patch)}
          />
        )}
        {(lights || []).map((light) => (
          <SceneLight
            key={light.id}
            light={light}
            lit={lampsLit}
            shown={!hidden[`light:${light.id}`]}
            selected={selected?.kind === "light" && selected.id === light.id}
            tool={tool}
            onSelect={() => onSelect({ kind: "light", id: light.id })}
            onCommit={(patch) => onCommitLight(light.id, patch)}
          />
        ))}
        {gas && !hidden.domain && (
          <GasDomain
            gas={gas}
            selected={selected?.kind === "domain"}
            tool={tool}
            onSelect={() => onSelect({ kind: "domain" })}
            onCommit={onCommitGas}
          />
        )}
        {!hidden.origin && (
          <OriginMarker
            origin={origin}
            selected={selected?.kind === "origin"}
            tool={tool}
            onSelect={() => onSelect({ kind: "origin" })}
            onCommit={onCommitOrigin}
          />
        )}
        {cameras.filter((entry) => !hidden[`camera:${entry.viewpoint}`]).map((entry) => (
          <BakeCamera
            key={entry.viewpoint}
            pose={entry.pose}
            domain={shell.domain}
            label={entry.pose.fallback ? `${entry.viewpoint}（借用侧视）` : entry.viewpoint}
            selected={selected?.kind === "camera" && selected.viewpoint === entry.viewpoint}
            part={selected?.kind === "camera" && selected.viewpoint === entry.viewpoint ? selected.part : "body"}
            tool={tool}
            onSelect={(part) => onSelect({ kind: "camera", viewpoint: entry.viewpoint, part })}
            onLook={() => onLook(entry.pose)}
            onCommit={(patch) => onCommitCamera(entry.viewpoint, patch)}
          />
        ))}
        <PlacementGhost ghost={ghost} />
        <OrbitControls
          makeDefault
          target={shot.target}
          maxPolarAngle={Math.PI * 0.72}
          mouseButtons={{
            LEFT: THREE.MOUSE.ROTATE,
            MIDDLE: THREE.MOUSE.ROTATE,
            RIGHT: THREE.MOUSE.PAN,
          }}
        />
        <GizmoHelper alignment="top-right" margin={[72, 72]}>
          <GizmoViewport />
        </GizmoHelper>
        <ViewBridge aim={aim} />
        <DropBridge onHover={onHover} onDrop={onDrop} />
        <CanvasBridge onCanvas={onCanvas} />
        {items.map((item, index) => !hidden[`piece:${index}`] && (
          <Package
            key={`${item.asset_id}-${index}`}
            item={item}
            asset={assetRecord(assets, item)}
            selected={selected?.kind === "piece" && selected.index === index}
            mode={tool}
            issues={issues[index] || []}
            onSelect={() => onSelect({ kind: "piece", index })}
            onCommit={(patch) => onCommit(index, patch)}
          />
        ))}
      </Canvas>
    </div>
  );
}

function FuelSelect({ label, value, fuels, onChange }) {
  const options = fuels?.length ? fuels : ["木材", "汽油", "塑料", "植被", "化学品"];
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <select value={value || "木材"} onChange={(event) => onChange(event.target.value)}>
        {options.map((fuel) => <option key={fuel} value={fuel}>{fuel}</option>)}
        {value && !options.includes(value) && <option value={value}>{value}</option>}
      </select>
    </label>
  );
}

function MeterField({ label, value, onCommit, compact = false }) {
  const [text, setText] = useState(() => formatMeter(value));
  const focused = useRef(false);
  useEffect(() => {
    if (!focused.current) setText(formatMeter(value));
  }, [value]);
  const commit = () => {
    const next = Number(text);
    if (!Number.isFinite(next)) {
      setText(formatMeter(value));
      return;
    }
    if (Math.abs(next - Number(value)) < 1e-6) return;
    if (onCommit(next) === false) setText(formatMeter(value));
  };
  return (
    <label className={compact ? "be-num" : "field"}>
      <span className={compact ? "be-num-label" : "field-label"}>{label}</span>
      <input
        inputMode="decimal"
        value={text}
        onFocus={() => { focused.current = true; }}
        onBlur={() => { focused.current = false; commit(); }}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") event.currentTarget.blur();
        }}
      />
    </label>
  );
}

function SliderField({ label, value, range, step, onCommit }) {
  const [low, high] = range || [0, 1];
  return (
    <div className="be-slider">
      <input
        type="range"
        min={low}
        max={high}
        step={step}
        value={Number(value)}
        aria-label={label}
        onChange={(event) => onCommit(Number(event.target.value))}
      />
      <MeterField compact label={label} value={value} onCommit={(next) => {
        if (next < low || next > high) return false;
        onCommit(next);
        return true;
      }} />
    </div>
  );
}

function formatCells(cells) {
  const value = Number(cells);
  if (!Number.isFinite(value)) return "";
  return value >= 10000 ? `${Math.round(value / 1000) / 10} 万格` : `${value} 格`;
}

function VecFields({ title, labels, value, onCommit }) {
  if (!value || value.length < labels.length) return null;
  return (
    <div className="be-vec">
      {title && <span className="be-vec-title">{title}</span>}
      <div className="be-vec-fields">
      {labels.map((label, index) => (
        <MeterField
          key={label}
          compact
          label={label}
          value={value[index]}
          onCommit={(next) => {
            const point = value.map(Number);
            point[index] = next;
            return onCommit(point);
          }}
        />
      ))}
      </div>
    </div>
  );
}

function shellRecord(shell, id) {
  if (!shell?.domain) return null;
  if (id === "floor") {
    const [length, width] = shell.domain;
    return { id: "floor", label: "地板", kind: "inert", size: [length, width, 0.02], center: [length / 2, width / 2, 0] };
  }
  const obstacle = (shell.obstacles || []).find((item) => item.name === id);
  if (!obstacle?.xb) return null;
  const xb = obstacle.xb;
  return {
    id,
    label: obstacle.label || SHELL_LABEL[id] || id,
    kind: obstacle.kind || "inert",
    size: [xb[1] - xb[0], xb[3] - xb[2], xb[5] - xb[4]],
    center: [(xb[0] + xb[1]) / 2, (xb[2] + xb[3]) / 2, (xb[4] + xb[5]) / 2],
  };
}

function issueNote(code, domainCap) {
  if (code === "shaft") return "挡住火柱。提交时会被拒绝。";
  if (code === "outside") return "超出场景。提交时会被拒绝。";
  if (code === "far") return "气体域盖不住。火焰只在气体域里面烧。";
  return "";
}

/** Shell cameras for the chosen viewpoints, in catalog order. */
export function viewCameras(shell, viewpoints) {
  const cameras = shell?.cameras || {};
  return (viewpoints || []).filter((view) => cameras[view]).map((view) => ({ viewpoint: view, pose: cameras[view] }));
}

/**
 * The 3D layout editor. It fills its container; the page decides what sits
 * around it. ``header`` goes at the start of the toolbar and ``side`` becomes
 * a third column, so the workbench can put generation next to the layout.
 */
function shownCameras(shell, viewpoints, overrides, preview) {
  return (viewpoints || []).filter((view) => shell?.cameras?.[view] || overrides?.[view] || preview?.cameras?.[view]).map((view) => {
    const custom = overrides?.[view];
    const resolved = preview?.cameras?.[view];
    const base = shell?.cameras?.[view];
    if (custom && !resolved?.inside_solid) {
      return {
        viewpoint: view,
        pose: {
          ...(resolved || base || {}),
          location: custom.location,
          target: custom.target,
          focal_mm: custom.focal_mm,
          overridden: true,
          fallback: false,
          blocked: Boolean(resolved?.blocked),
          inside_solid: false,
        },
      };
    }
    return { viewpoint: view, pose: resolved || base };
  }).filter((entry) => entry.pose?.location);
}

// Editor layout preferences are per-viewer conveniences. Storage can be
// missing or blocked, so every read and write falls back quietly.
function readPref(key, fallback) {
  try {
    const raw = window.localStorage.getItem(key);
    if (raw == null) return fallback;
    return JSON.parse(raw) ?? fallback;
  } catch {
    return fallback;
  }
}

function usePref(key, fallback) {
  const [value, setValue] = useState(() => readPref(key, fallback));
  useEffect(() => {
    try { window.localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage blocked */ }
  }, [key, value]);
  return [value, setValue];
}

function useDrag(onMove) {
  const latest = useRef(onMove);
  latest.current = onMove;
  return (event) => {
    event.preventDefault();
    const move = (next) => latest.current(next);
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      document.body.classList.remove("be-dragging");
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    document.body.classList.add("be-dragging");
  };
}

export function Panel({ title, hint, actions, defaultOpen = true, children }) {
  const [open, setOpen] = useState(defaultOpen);
  const lines = [].concat(hint || []).filter(Boolean);
  return (
    <section className={open ? "be-panel open" : "be-panel"}>
      <header className="be-panel-head">
        <button type="button" className="be-panel-toggle" aria-expanded={open} onClick={() => setOpen((value) => !value)}>
          {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          <span>{title}</span>
        </button>
        {actions}
      </header>
      {open && (
        <div className="be-panel-body">
          {children}
          {lines.length > 0 && (
            <details className="be-hint">
              <summary>说明</summary>
              {lines.map((line) => <p key={line}>{line}</p>)}
            </details>
          )}
        </div>
      )}
    </section>
  );
}

/** Visibility key of a selection, matching the viewport's hidden map. */
function hiddenKey(selection) {
  switch (selection?.kind) {
    case "shell": return `shell:${selection.id}`;
    case "piece": return `piece:${selection.index}`;
    case "light": return `light:${selection.id}`;
    case "camera": return `camera:${selection.viewpoint}`;
    case "origin": return "origin";
    case "domain": return "domain";
    default: return null;
  }
}

function OutlinerGroup({ title, icon: Icon, count, open, onToggle, action, children }) {
  return (
    <div className="be-ol-group">
      <div className="be-ol-head">
        <button type="button" className="be-ol-toggle" aria-expanded={open} onClick={onToggle}>
          {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          <Icon size={13} />
          <span>{title}</span>
          <small>{count}</small>
        </button>
        {action}
      </div>
      {open && <div className="be-ol-rows">{children}</div>}
    </div>
  );
}

function OutlinerRow({ icon: Icon, thumb, label, tag, warn, picked, dim, title, visible = true, onToggleVisible, onClick, onDoubleClick }) {
  return (
    <div className="be-ol-line">
      <button
        type="button"
        className={`be-ol-row${picked ? " selected" : ""}${dim || !visible ? " dim" : ""}`}
        title={title}
        onClick={onClick}
        onDoubleClick={onDoubleClick}
      >
        {thumb || (Icon && <Icon size={13} className="be-ol-icon" />)}
        <span className="be-ol-name">{label}</span>
        {warn && <TriangleAlert size={12} className="be-ol-warn" />}
        {tag && <small className="be-ol-tag">{tag}</small>}
      </button>
      {onToggleVisible && (
        <button
          type="button"
          className={visible ? "be-ol-eye" : "be-ol-eye off"}
          title={visible ? "在视口隐藏" : "在视口显示"}
          aria-label={`${visible ? "隐藏" : "显示"}${label}`}
          aria-pressed={!visible}
          onClick={onToggleVisible}
        >
          {visible ? <Eye size={12} /> : <EyeOff size={12} />}
        </button>
      )}
    </div>
  );
}

const VIEWS = [["perspective", "透视"], ["top", "顶视"], ["side", "侧视"]];
const KEY_HINT = "单击选中 · 双击相机进入视角 · G 移动 · R 转向 · S 缩放气体域或结构 · F 对准 · 方向键 0.1 m · Shift+↑↓ 升降 · ⌫ 删除 · ⌘S 保存";

export function SceneWorkstation({
  scenario, scenarios, shell, items: rawItems, assets, fuels, selected, tool,
  domainCap, maxPlacements, originFuel, originOverride, cameraOverrides, gasOverride, preview, viewpoints, header, headerEnd, tabs, activeTab, onTab,
  lights: lightsOverride, automaticLights, maxLights, gasLimits, simulation, onSimulation,
  structures, domainOverride, maxStructures,
  onApplyPreset, onStructures, onDomain, onOriginFuel, onOrigin, onCameras, onGas, onLights, onTool, onSelect, onCommit, onAdd, onRemove, onDuplicate, onUpload, onRestore, onCanvas, onSave,
}) {
  const items = useMemo(() => placementsWithSize(rawItems, assets), [rawItems, assets]);
  const rootRef = useRef(null);
  const dragAsset = useRef(null);
  const suppressClick = useRef(false);
  const [note, setNote] = useState("");
  const [aim, setAim] = useState(null);
  const [ghost, setGhost] = useState(null);
  const [confirmRestore, setConfirmRestore] = useState(false);
  const [confirmPreset, setConfirmPreset] = useState("");
  const [addingLight, setAddingLight] = useState(false);
  const bodyRef = useRef(null);
  const sideRef = useRef(null);
  const lightMenu = useRef(null);
  const [sideWidth, setSideWidth] = usePref("firephantom.editor.side", 340);
  const [outlinerShare, setOutlinerShare] = usePref("firephantom.editor.outliner", 0.42);
  const [shelfOpen, setShelfOpen] = usePref("firephantom.editor.shelf", true);
  const [shelfQuery, setShelfQuery] = useState("");
  const [shelfGroup, setShelfGroup] = useState("全部");
  const [folded, setFolded] = usePref("firephantom.editor.folded", {});
  const [shadingPref, setShading] = usePref("firephantom.editor.shading", "solid");
  const shading = SHADING.some(([id]) => id === shadingPref) ? shadingPref : "solid";
  const [sceneLights, setSceneLights] = usePref("firephantom.editor.sceneLights", false);
  // Viewport-only visibility, per scenario. The bake never reads it.
  const [hiddenByScenario, setHiddenByScenario] = usePref("firephantom.editor.hidden", {});
  const hidden = hiddenByScenario?.[scenario] || {};
  const setHidden = (update) => setHiddenByScenario((all) => {
    const next = update(all?.[scenario] || {});
    return { ...all, [scenario]: next };
  });
  const [ownTab, setOwnTab] = useState("object");
  const tab = activeTab || ownTab;
  const setTab = (id) => (onTab ? onTab(id) : setOwnTab(id));
  const dragSide = useDrag((event) => {
    const rect = bodyRef.current?.getBoundingClientRect();
    if (!rect) return;
    setSideWidth(Math.round(Math.min(520, Math.max(260, rect.right - event.clientX))));
  });
  const dragSplit = useDrag((event) => {
    const rect = sideRef.current?.getBoundingClientRect();
    if (!rect?.height) return;
    setOutlinerShare(Math.min(0.78, Math.max(0.15, (event.clientY - rect.top) / rect.height)));
  });
  useEffect(() => {
    if (!addingLight) return undefined;
    const close = (event) => {
      if (!lightMenu.current?.contains(event.target)) setAddingLight(false);
    };
    window.addEventListener("pointerdown", close);
    return () => window.removeEventListener("pointerdown", close);
  }, [addingLight]);
  const item = selected?.kind === "piece" ? items[selected.index] : null;
  const full = items.length >= maxPlacements;
  const cameras = shownCameras(shell, viewpoints, cameraOverrides, preview);
  const site = Array.isArray(domainOverride) && domainOverride.length === 3 ? domainOverride : shell?.domain;
  const presetBoxes = useMemo(() => boxesFromObstacles(shell), [shell]);
  const boxes = structures == null ? presetBoxes : structures;
  const origin = originOverride || shell?.origin || null;
  const liveShell = shell && site && origin ? { ...shell, domain: site, origin, obstacles: boxes.map(boxToObstacle) } : shell;
  const structureLimit = maxStructures || 24;
  const gas = gasOverride || preview?.gas_domain || null;
  const gasBounds = boundsOfGas(gas);
  const lights = lightsOverride == null ? (automaticLights || []) : lightsOverride;
  const lightsAutomatic = lightsOverride == null;
  const lightLimit = maxLights || 8;
  const limits = gasLimits || {};
  const maxResolution = limits.resolution?.[1] || 128;
  const lookIndex = useRef(0);
  useEffect(() => {
    if (selected?.kind === "camera" && !cameras.some((entry) => entry.viewpoint === selected.viewpoint)) onSelect(null);
  }, [cameras, selected, onSelect]);
  useEffect(() => {
    if (selected?.kind === "light" && !lights.some((entry) => entry.id === selected.id)) onSelect(null);
  }, [lights, selected, onSelect]);
  useEffect(() => {
    if (selected?.kind === "shell" && selected.id !== "floor" && !boxes.some((box) => box.id === selected.id)) onSelect(null);
  }, [boxes, selected, onSelect]);
  useEffect(() => { setConfirmRestore(false); setConfirmPreset(""); }, [scenario]);
  const domainKey = site ? site.join(",") : "";
  useEffect(() => {
    if (!site) return;
    setAim({ ...cameraFor({ domain: site }, "perspective"), token: `site:${scenario}:${domainKey}` });
  }, [scenario, domainKey, site]);
  useEffect(() => {
    const end = () => setGhost(null);
    window.addEventListener("dragend", end);
    return () => window.removeEventListener("dragend", end);
  }, []);
  const aimAt = (shot) => {
    if (!shot) return;
    setAim({ ...shot, token: Date.now() });
  };
  const add = (asset, point) => {
    const message = onAdd(asset, point);
    setNote(message || "");
  };
  const shelfGroups = useMemo(() => {
    const order = ["家具", "树", "灌木与草", "石头", "倒木", "容器", "我的上传", "其他"];
    const present = new Set((assets || []).map((asset) => (String(asset.asset_id || "").startsWith("upload:") ? "我的上传" : (asset.group || "其他"))));
    return ["全部", ...order.filter((name) => present.has(name))];
  }, [assets]);
  const shelfAssets = useMemo(() => {
    const needle = shelfQuery.trim().toLowerCase();
    return (assets || []).filter((asset) => {
      const group = String(asset.asset_id || "").startsWith("upload:") ? "我的上传" : (asset.group || "其他");
      if (shelfGroup !== "全部" && group !== shelfGroup) return false;
      if (!needle) return true;
      return `${asset.label || ""} ${group} ${asset.fuel_type || ""}`.toLowerCase().includes(needle);
    });
  }, [assets, shelfGroup, shelfQuery]);
  const hoverGhost = (point, transfer) => {
    if (!point) {
      setGhost(null);
      return;
    }
    const asset = dragAsset.current;
    if (asset?.size) {
      const error = spotError(liveShell, items, asset.size, point.x, point.y);
      setGhost((current) => {
        if (current?.asset === asset && current.error === error && Math.hypot(current.x - point.x, current.y - point.y) < 0.03) return current;
        return { asset, x: point.x, y: point.y, error };
      });
      return;
    }
    if (Array.from(transfer?.types || []).includes("Files")) {
      setGhost((current) => {
        if (current?.file && Math.hypot(current.x - point.x, current.y - point.y) < 0.03) return current;
        return { file: true, x: point.x, y: point.y };
      });
    }
  };
  const dropGhost = async (point, transfer) => {
    const asset = dragAsset.current;
    dragAsset.current = null;
    setGhost(null);
    if (!point) {
      setNote("放到地面上");
      return;
    }
    const file = Array.from(transfer?.files || []).find((entry) => entry.name.toLowerCase().endsWith(".glb"));
    if (file) {
      setNote("");
      try {
        const uploaded = await onUpload(file);
        add(uploaded, point);
      } catch (error) {
        setNote(error.message || "上传失败");
      }
      return;
    }
    const assetId = transfer?.getData?.("application/x-firephantom-asset") || transfer?.getData?.("text/plain") || asset?.asset_id;
    const found = (assets || []).find((entry) => entry.asset_id === assetId) || asset;
    if (found) add(found, point);
  };
  const poseOf = (viewpoint) => cameras.find((entry) => entry.viewpoint === viewpoint)?.pose;
  const commitOrigin = (point) => {
    if (!site || !shell?.origin) return;
    const clamped = clampOrigin(point, site);
    onOrigin(sameVec(clamped, shell.origin) ? null : clamped);
  };
  const commitCamera = (viewpoint, patch) => {
    const current = cameraOverrides?.[viewpoint] || poseOf(viewpoint);
    if (!current?.location || !site) return;
    const next = {
      location: patch.location ? clampCameraPoint(patch.location, site) : current.location,
      target: patch.target || current.target,
      focal_mm: patch.focal_mm == null ? current.focal_mm : Math.min(120, Math.max(12, Number(patch.focal_mm))),
    };
    if (!Number.isFinite(next.focal_mm)) return false;
    const apart = Math.hypot(
      next.location[0] - next.target[0],
      next.location[1] - next.target[1],
      next.location[2] - next.target[2],
    );
    if (!(apart >= 0.05)) return false;
    if (patch.location && pointInSolid(next.location, liveShell)) {
      setNote("相机在墙或天花板里面");
      return false;
    }
    const auto = preview?.automatic_cameras?.[viewpoint] || shell.cameras?.[viewpoint];
    const map = { ...(cameraOverrides || {}) };
    if (samePose(next, auto)) delete map[viewpoint];
    else map[viewpoint] = next;
    onCameras(map);
    return true;
  };
  const baseLights = () => (lightsOverride == null ? (automaticLights || []).map((light) => ({ ...light })) : lightsOverride.map((light) => ({ ...light })));
  const commitLights = (next) => {
    const auto = automaticLights || [];
    onLights(auto.length && sameLights(next, auto) ? null : next);
  };
  const updateLight = (id, patch) => {
    const current = baseLights().find((light) => light.id === id);
    if (!current) return;
    const nextPatch = { ...patch };
    if (nextPatch.location && site) nextPatch.location = clampLightPoint(nextPatch.location, site);
    const kind = nextPatch.type || current.type;
    if (nextPatch.type === "SUN" || nextPatch.energy != null) {
      const energy = nextPatch.energy != null ? nextPatch.energy : current.energy;
      const high = kind === "SUN" ? 10 : 2000;
      nextPatch.energy = round4(Math.min(high, Math.max(0, Number(energy) || 0)));
    }
    if (nextPatch.size != null) nextPatch.size = round4(Math.min(5, Math.max(0.01, Number(nextPatch.size) || 0.01)));
    if (nextPatch.size_y != null) nextPatch.size_y = round4(Math.min(5, Math.max(0.01, Number(nextPatch.size_y) || 0.01)));
    commitLights(baseLights().map((light) => (light.id === id ? { ...light, ...nextPatch } : light)));
  };
  const removeLight = (id) => {
    commitLights(baseLights().filter((light) => light.id !== id));
    if (selected?.kind === "light" && selected.id === id) select(null);
  };
  const addLight = (type) => {
    const base = baseLights();
    if (lightsAutomatic && !base.length) return;
    if (base.length >= lightLimit || !origin || !site) return;
    const count = base.filter((light) => light.type === type).length;
    const label = LIGHT_LABEL[type] || "灯";
    const sun = type === "SUN";
    const light = {
      id: `L${Date.now().toString(36)}`,
      name: count ? `${label} ${count + 1}` : label,
      type,
      color: sun ? [1, 0.96, 0.9] : [1, 0.95, 0.85],
      energy: sun ? 2.2 : 200,
      size: type === "AREA" ? 0.6 : sun ? 0.1 : 0.4,
      size_y: type === "AREA" ? 0.4 : 1,
      spot_size: 0.8,
      spot_blend: 0.15,
      location: clampLightPoint([origin[0] + 0.6, origin[1], Math.min((origin[2] || 0) + 1.6, site[2] - 0.2)], site),
      rotation: type === "POINT" ? [0, 0, 0] : [1, 0, 0],
      enabled: true,
    };
    commitLights([...base, light]);
    select({ kind: "light", id: light.id });
    setAddingLight(false);
  };
  const commitGas = (patch) => {
    const base = gasOverride || preview?.automatic_gas || preview?.gas_domain;
    if (!base || !site) return;
    const next = clampGas({
      location: patch.location || base.location,
      scale: patch.scale || base.scale,
      resolution: patch.resolution == null ? base.resolution : patch.resolution,
    }, site, maxResolution);
    if (patch.scale && !sameVec(patch.scale, next.scale)) {
      setNote(`气体域每边至少 0.5 m，也不能超出场景，已调整为 ${next.scale.map((edge) => formatMeter(edge * 2)).join(" × ")} m`);
    } else if (patch.location && !sameVec(patch.location, next.location)) {
      setNote("气体域不能超出场景，已移回场景内");
    } else if (patch.resolution != null && Number(patch.resolution) !== next.resolution) {
      setNote(`分辨率要在 16–${maxResolution} 之间，已调整为 ${next.resolution}`);
    }
    const auto = preview?.automatic_gas;
    onGas(auto && sameGas(next, auto) ? null : next);
  };
  const fitGas = () => {
    const auto = preview?.automatic_gas || preview?.gas_domain;
    if (!auto || !site) return;
    const fitted = fitGasToItems(gasOverride || auto, items, site, maxResolution);
    if (!fitted) return;
    const autoGas = preview?.automatic_gas;
    onGas(autoGas && sameGas(fitted.gas, autoGas) ? null : fitted.gas);
    setNote(fitted.missed.length
      ? `${fitted.missed.map((entry) => entry.name || entry.asset_id).join("、")} 超出了场景，盒子包不到`
      : "");
  };
  // Walls and floor are pulled back by the server, which then notes the snap.
  const fillGas = () => {
    if (!site) return;
    const [length, width, height] = site;
    commitGas({ location: [length / 2, width / 2, height / 2], scale: [length / 2, width / 2, height / 2] });
  };
  const storedBoxes = () => (structures == null ? presetBoxes : structures).map((box) => ({ ...box, center: [...box.center], size: [...box.size] }));
  const commitStructures = (next) => {
    onStructures?.(collapseStructures(presetBoxes, next));
  };
  const updateStructure = (id, patch) => {
    const current = storedBoxes().find((box) => box.id === id);
    if (!current || !site) return false;
    let next = { ...current, ...patch };
    if (patch.name != null) next.name = String(patch.name).slice(0, 40);
    if (patch.center || patch.size) {
      const clamped = clampStructure(next, site);
      if (patch.center && !sameVec(patch.center, clamped.center, 1e-4)) setNote(`${clamped.name || "结构"}超出场地，已移回场地内`);
      else if (patch.size && !sameVec(patch.size, clamped.size, 1e-4)) {
        const tooSmall = patch.size.some((edge) => Number(edge) < 0.05);
        const tooBig = patch.size.some((edge, index) => Number(edge) > site[index] + 1e-4);
        setNote(tooSmall ? "结构每边至少 0.05 m，已调整" : tooBig ? "结构的边不能比场地更长，已调整" : `${clamped.name || "结构"}超出场地，已移回场地内`);
      }
      next = clamped;
    }
    commitStructures(storedBoxes().map((box) => (box.id === id ? next : box)));
    return true;
  };
  const removeStructure = (id) => {
    commitStructures(storedBoxes().filter((box) => box.id !== id));
    if (selected?.kind === "shell" && selected.id === id) select(null);
  };
  const addStructure = () => {
    const base = storedBoxes();
    if (!site || !origin) return;
    if (base.length >= structureLimit) {
      setNote(`最多 ${structureLimit} 块结构`);
      return;
    }
    const span = [0, 1, 2].map((index) => round4(Math.min(index === 2 ? 0.2 : 1, site[index])));
    let center = [site[0] / 2, site[1] / 2, span[2] / 2];
    const half = shell?.shaft_half_m || 1;
    const overlapsShaft = (point) => {
      const x0 = point[0] - span[0] / 2;
      const x1 = point[0] + span[0] / 2;
      const y0 = point[1] - span[1] / 2;
      const y1 = point[1] + span[1] / 2;
      return x0 < origin[0] + half && x1 > origin[0] - half && y0 < origin[1] + half && y1 > origin[1] - half;
    };
    if (overlapsShaft(center)) center = [site[0] - span[0] / 2, site[1] / 2, span[2] / 2];
    const box = clampStructure({
      id: nextStructureId(base),
      name: "结构",
      role: "inert",
      center,
      size: span,
    }, site);
    commitStructures([...base, box]);
    select({ kind: "shell", id: box.id });
  };
  const commitDomain = (point) => {
    if (!shell?.domain || !point) return false;
    const next = point.map((value) => round4(Number(value)));
    if (next.some((value) => !Number.isFinite(value) || value < 1 || value > 80)) {
      setNote("场地每边要在 1 到 80 米");
      return false;
    }
    if (storedBoxes().some((box) => !structureInside(box, next))) {
      setNote("有结构超出这块场地");
      return false;
    }
    if (items.some((item) => item?.size?.length === 3 && !insideDomain(item, next))) {
      setNote("有燃料超出这块场地");
      return false;
    }
    onDomain?.(sameVec(next, shell.domain) ? null : next);
    setNote("");
    return true;
  };
  const simDefaults = limits.defaults;
  const sim = { ...simDefaults, ...(simulation || preview?.simulation || {}) };
  const setSim = (patch) => {
    const next = { ...sim, ...patch };
    const same = Object.keys(simDefaults).every((key) => Math.abs(Number(next[key]) - Number(simDefaults[key])) < 1e-6);
    onSimulation?.(same ? null : next);
  };
  const scalable = (next) => next?.kind === "domain" || (next?.kind === "shell" && next.id !== "floor");
  const select = (next) => {
    if (tool === "scale" && !scalable(next)) onTool("translate");
    if (next && tab !== "object") setTab("object");
    onSelect(next);
  };
  const visible = (key) => !hidden[key];
  const toggleVisible = (key) => (event) => {
    event.stopPropagation();
    const hiding = !hidden[key];
    setHidden((current) => {
      const next = { ...current };
      if (hiding) next[key] = true;
      else delete next[key];
      return next;
    });
    // A gizmo on an invisible object would still catch drags.
    if (hiding && selected && hiddenKey(selected) === key) onSelect(null);
  };
  const toggleShell = (obstacle) => (event) => {
    event.stopPropagation();
    const key = `shell:${obstacle.name}`;
    const shown = shellEyeOn(obstacle, hidden);
    setHidden((current) => {
      const next = { ...current };
      if (shown) next[key] = true;
      else if (isCeiling(obstacle)) next[key] = false;
      else delete next[key];
      return next;
    });
    if (shown && selected?.kind === "shell" && selected.id === obstacle.name) onSelect(null);
  };
  const setShellVisible = (show) => {
    setHidden((current) => {
      const next = { ...current };
      if (show) delete next["shell:floor"];
      else next["shell:floor"] = true;
      for (const box of boxes) {
        const key = `shell:${box.id}`;
        const ceiling = isCeiling(boxToObstacle(box));
        if (!show) next[key] = true;
        else if (ceiling) next[key] = false;
        else delete next[key];
      }
      return next;
    });
    if (!show && selected?.kind === "shell") onSelect(null);
  };
  // Hidden pieces are keyed by index, so later ones shift down on removal.
  const removePiece = (index) => {
    setHidden((current) => {
      const next = {};
      for (const [key, value] of Object.entries(current)) {
        const match = /^piece:(\d+)$/.exec(key);
        if (!match) next[key] = value;
        else if (Number(match[1]) < index) next[key] = value;
        else if (Number(match[1]) > index) next[`piece:${Number(match[1]) - 1}`] = value;
      }
      return next;
    });
    onRemove(index);
  };
  const enterView = (viewpoint) => {
    const pose = poseOf(viewpoint);
    if (!pose) return;
    select({ kind: "camera", viewpoint, part: "body" });
    const shot = bakeShot(shell, pose);
    if (shot) setAim({ ...shot, fov: focalFov(pose.focal_mm), token: Date.now() });
  };
  const frameSelection = () => {
    if (!shell) return;
    if (selected?.kind === "piece" && items[selected.index]) aimAt(frameShot(liveShell, items[selected.index]));
    else if (selected?.kind === "camera") {
      const pose = poseOf(selected.viewpoint);
      if (pose) aimAt({ ...bakeShot(shell, pose), fov: focalFov(pose.focal_mm) });
    } else if (selected?.kind === "domain" && gas) {
      aimAt(frameShot(liveShell, { x: gas.location[0], y: gas.location[1], z: gas.location[2], size: gas.scale.map((value) => value * 2) }));
    } else if (selected?.kind === "origin" && origin) {
      aimAt(frameShot(liveShell, { x: origin[0], y: origin[1], z: origin[2], size: [0.8, 0.8, 0.8] }));
    } else if (selected?.kind === "light") {
      const light = lights.find((entry) => entry.id === selected.id);
      if (light) aimAt(frameShot(liveShell, { x: light.location[0], y: light.location[1], z: light.location[2], size: [0.6, 0.6, 0.6] }));
    } else if (selected?.kind === "shell" && liveShell) {
      const record = shellRecord(liveShell, selected.id);
      if (record) aimAt(frameShot(liveShell, { x: record.center[0], y: record.center[1], z: record.center[2], size: record.size }));
    } else aimAt(frameShot(liveShell, null));
  };
  const lookThrough = () => {
    if (!cameras.length) return;
    const entry = cameras[lookIndex.current % cameras.length];
    lookIndex.current += 1;
    enterView(entry.viewpoint);
  };
  const nudgeSelection = (point, event) => {
    const step = 0.1;
    if (event.shiftKey && (event.key === "ArrowUp" || event.key === "ArrowDown")) {
      return [point[0], point[1], point[2] + (event.key === "ArrowUp" ? step : -step)];
    }
    const delta = { ArrowRight: [step, 0, 0], ArrowLeft: [-step, 0, 0], ArrowUp: [0, step, 0], ArrowDown: [0, -step, 0] }[event.key];
    if (!delta || event.shiftKey) return null;
    return [point[0] + delta[0], point[1] + delta[1], point[2] + delta[2]];
  };
  const onKey = (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s" && onSave) {
      event.preventDefault();
      onSave();
      return;
    }
    const tag = event.target?.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    if (event.key === "Escape") {
      select(null);
      return;
    }
    if (event.key === "Delete" || event.key === "Backspace") {
      if (selected?.kind === "light") {
        event.preventDefault();
        removeLight(selected.id);
        return;
      }
      if (selected?.kind === "shell" && selected.id !== "floor") {
        event.preventDefault();
        removeStructure(selected.id);
        return;
      }
      if (selected?.kind !== "piece" || !items[selected.index]) return;
      event.preventDefault();
      removePiece(selected.index);
      return;
    }
    if (!event.metaKey && !event.ctrlKey && !event.altKey) {
      const key = event.key.toLowerCase();
      if (key === "g") { event.preventDefault(); onTool("translate"); return; }
      if (key === "r") { event.preventDefault(); onTool("rotate"); return; }
      if (key === "s" && scalable(selected)) { event.preventDefault(); onTool("scale"); return; }
      if (key === "f" && shell) {
        event.preventDefault();
        frameSelection();
        return;
      }
    }
    if (selected?.kind === "piece" && items[selected.index]) {
      if (event.shiftKey && (event.key === "ArrowUp" || event.key === "ArrowDown")) {
        event.preventDefault();
        const step = event.key === "ArrowUp" ? 0.1 : -0.1;
        onCommit(selected.index, { z: Number(items[selected.index].z) + step });
        return;
      }
      const nudge = { ArrowRight: [0.1, 0], ArrowLeft: [-0.1, 0], ArrowUp: [0, 0.1], ArrowDown: [0, -0.1] }[event.key];
      if (!nudge) return;
      event.preventDefault();
      onCommit(selected.index, { x: items[selected.index].x + nudge[0], y: items[selected.index].y + nudge[1] });
      return;
    }
    const pickedLight = selected?.kind === "light" ? lights.find((entry) => entry.id === selected.id) : null;
    const pickedBox = selected?.kind === "shell" && selected.id !== "floor" ? boxes.find((box) => box.id === selected.id) : null;
    const point = selected?.kind === "origin" ? origin
      : selected?.kind === "domain" ? gas?.location
        : selected?.kind === "camera" ? (selected.part === "target" ? poseOf(selected.viewpoint)?.target : poseOf(selected.viewpoint)?.location)
          : pickedLight?.location
            || pickedBox?.center
            || null;
    const moved = point ? nudgeSelection(point, event) : null;
    if (!moved) return;
    event.preventDefault();
    if (selected.kind === "origin") commitOrigin(moved);
    else if (selected.kind === "domain") commitGas({ location: moved });
    else if (selected.kind === "shell") updateStructure(selected.id, { center: moved });
    else if (selected.kind === "light") updateLight(selected.id, { location: clampLightPoint(moved, site) });
    else if (selected.part === "target") commitCamera(selected.viewpoint, { target: moved });
    else commitCamera(selected.viewpoint, { location: moved });
  };
  const rejected = items.filter((entry) => pieceIssues(entry, items, liveShell, domainCap, gasBounds).some((code) => code === "shaft" || code === "outside")).length;
  const cameraPose = selected?.kind === "camera" ? poseOf(selected.viewpoint) : null;
  const pickedLight = selected?.kind === "light" ? lights.find((entry) => entry.id === selected.id) : null;
  const shellBox = selected?.kind === "shell" && selected.id !== "floor" ? boxes.find((box) => box.id === selected.id) : null;
  const floorInfo = selected?.kind === "shell" && selected.id === "floor" ? shellRecord(liveShell, "floor") : null;
  const editingObstacle = shellBox ? boxToObstacle(shellBox) : null;
  const editingId = editingObstacle && shellDrawn(editingObstacle, hidden, selected) ? shellBox.id : null;
  const flip = (key) => setFolded((current) => ({ ...current, [key]: !current?.[key] }));
  const isOpen = (key) => !folded?.[key];
  const restore = () => {
    if (!confirmRestore) { setConfirmRestore(true); return; }
    setConfirmRestore(false);
    onRestore();
  };
  const applyPreset = (name) => {
    if (confirmPreset !== name) {
      setConfirmPreset(name);
      setConfirmRestore(false);
      return;
    }
    setConfirmPreset("");
    onApplyPreset?.(name);
  };
  const lightBlocked = (lightsAutomatic && !lights.length) || lights.length >= lightLimit;
  const tabList = [
    { id: "object", label: "物体", icon: Box },
    { id: "scene", label: "场景", icon: Globe },
    ...(tabs || []),
  ];
  const currentTab = tabList.find((entry) => entry.id === tab) || tabList[0];
  const TabIcon = currentTab.icon;
  const objectTitle = item ? (item.name || item.asset_id)
    : pickedLight ? (pickedLight.name || LIGHT_LABEL[pickedLight.type])
      : cameraPose ? selected.viewpoint
        : selected?.kind === "origin" ? "起火点"
          : selected?.kind === "domain" ? "气体域"
            : shellBox?.name || floorInfo?.label || "";
  const itemIssues = item ? pieceIssues(item, items, liveShell, domainCap, gasBounds) : [];
  const uncovered = selected?.kind === "domain"
    ? items.map((entry, index) => ({ entry, index })).filter(({ entry }) => pieceIssues(entry, items, liveShell, domainCap, gasBounds).includes("far"))
    : [];
  const gasCell = gasCellCm(gas);
  const estimate = preview?.bake_estimate || null;

  const shadingSwitch = (
    <div className="be-seg" role="radiogroup" aria-label="视口着色">
      {SHADING.map(([id, label]) => (
        <button type="button" role="radio" key={id} aria-checked={shading === id} className={shading === id ? "active" : ""} onClick={() => setShading(id)}>{label}</button>
      ))}
    </div>
  );
  const sceneLightsCheck = (
    <label className="be-check" title="材质预览里也打开场景里的灯">
      <input type="checkbox" checked={Boolean(sceneLights)} onChange={(event) => setSceneLights(event.target.checked)} />
      <span>场景灯光</span>
    </label>
  );

  const objectProps = (
    <>
      {selected?.kind === "origin" && origin && (
        <>
          <Panel title="变换" hint="火柱、自动气体域和没改过的相机会跟着这个点。房间的燃料床、林区植被和工业区油池外壳不动。">
            <VecFields title="位置" labels={["X", "Y", "Z"]} value={origin} onCommit={commitOrigin} />
            {originOverride && <button type="button" className="secondary-button" onClick={() => onOrigin(null)}><RotateCcw size={13} />恢复自动</button>}
          </Panel>
          <Panel title="材料">
            <FuelSelect label="起火点材料" value={originFuel} fuels={fuels} onChange={onOriginFuel} />
          </Panel>
        </>
      )}
      {selected?.kind === "domain" && (
        <>
          <Panel
            title="气体域"
            hint={[
              `自动的盒子包住起火点上方的火柱，再往起火点 ${domainCap} m 内的燃料长，每边最多 ${domainCap} m，烘焙最快。`,
              "自定义的盒子可以铺满整个场景，只受算力上限约束。",
              "贴着地面或整面墙时，盒子会自动让开大约 5 cm，否则 Cycles 渲染不出烟。",
            ]}
          >
            {gas ? (
              <>
                <p className="field-hint">盒子以外的燃料不会冒烟起火。盒子越大、越细、模拟越久，烘焙越慢。</p>
                {estimate && (
                  <div className={estimate.over ? "be-budget over" : "be-budget"}>
                    <div className="be-budget-bar"><i style={{ width: `${Math.min(100, (estimate.seconds / estimate.limit) * 100)}%` }} /></div>
                    <span>
                      预计烘焙约 {formatDuration(estimate.seconds)} · {formatCells(estimate.cells)}
                      <small> / 上限 {formatDuration(estimate.limit)}</small>
                    </span>
                    {estimate.over && <p>超出上限，提交会被拒绝。缩小盒子、降低分辨率或噪声、缩短预热，或在“渲染”页减少帧数、视角和天气组合。</p>}
                  </div>
                )}
                <div className="field">
                  <span className="field-label">状态</span>
                  <span className="readonly-value">{gasOverride ? "自定义" : "自动 · 跟着起火点和附近燃料"}</span>
                </div>
                <div className="panel-actions">
                  <button type="button" className="secondary-button" onClick={fillGas}><Box size={13} />铺满场景</button>
                  {gasOverride && <button type="button" className="secondary-button" onClick={() => { onGas(null); setNote(""); }}><RotateCcw size={13} />恢复自动</button>}
                </div>
                {uncovered.length > 0 ? (
                  <div className="stage-note be-gas-cover">
                    <p>{uncovered.length} 件燃料在盒子外面，不会烧起来：</p>
                    <p>
                      {uncovered.map(({ entry, index }, order) => (
                        <React.Fragment key={index}>
                          {order > 0 && "、"}
                          <button type="button" className="link-button" onClick={() => select({ kind: "piece", index })}>{entry.name || entry.asset_id}</button>
                        </React.Fragment>
                      ))}
                    </p>
                    <button type="button" className="secondary-button" onClick={fitGas}><Box size={13} />包住所有燃料</button>
                  </div>
                ) : items.length > 0 && <p className="field-hint">所有燃料都在盒子里。</p>}
                <VecFields title="中心" labels={["X", "Y", "Z"]} value={gas.location} onCommit={(point) => commitGas({ location: point })} />
                <VecFields title="尺寸 m" labels={["长", "宽", "高"]} value={gas.scale.map((edge) => edge * 2)} onCommit={(point) => commitGas({ scale: point.map((edge) => edge / 2) })} />
                <p className="field-hint">视口里选中后，按 G 拖动，按 S 拉伸。</p>
              </>
            ) : <p className="field-hint">气体域还在计算。</p>}
          </Panel>
          {gas && (
            <Panel title="精度" hint={["分辨率是盒子最长边分成几格。格越细，火焰边缘越细，烘焙时间大约按格数的立方增长。", "噪声细节在网格上再叠一层更细的湍流，×3 比 ×2 慢不少。"]}>
              <div className="field">
                <span className="field-label">分辨率</span>
                <div className="be-seg wide" role="radiogroup" aria-label="气体域分辨率">
                  {GAS_STEPS.map(([value, label]) => (
                    <button
                      type="button"
                      role="radio"
                      key={value}
                      aria-checked={Number(gas.resolution) === value}
                      className={Number(gas.resolution) === value ? "active" : ""}
                      title={`${value} 格`}
                      onClick={() => commitGas({ resolution: value })}
                    >{label}</button>
                  ))}
                </div>
              </div>
              <MeterField label={`格数 16–${maxResolution}`} value={gas.resolution} onCommit={(value) => commitGas({ resolution: value })} />
              {gasCell != null && <p className="field-hint">每格约 {gasCell} cm。</p>}
              <div className="field">
                <span className="field-label">噪声细节</span>
                <div className="be-seg wide" role="radiogroup" aria-label="噪声细节">
                  {NOISE_STEPS.map(([value, label]) => (
                    <button type="button" role="radio" key={value} aria-checked={Number(sim.noise) === value} className={Number(sim.noise) === value ? "active" : ""} onClick={() => setSim({ noise: value })}>{label}</button>
                  ))}
                </div>
              </div>
            </Panel>
          )}
          {gas && (
            <Panel title="时长" hint="出图前先模拟这么久，火和烟有时间长大、铺开。远处燃料仍在出图开始后的半秒内逐个点着。">
              <SliderField label="预热 s" value={sim.warmup_s} range={limits.ranges?.warmup_s} step={0.5} onCommit={(value) => setSim({ warmup_s: value })} />
            </Panel>
          )}
          {gas && (
            <Panel
              title="火焰外观"
              hint={["涡度：烟和火卷得多乱。燃烧速度：燃料多快烧完，越快火越短。", "火焰温度：越高越亮、越偏白。烟量：乘在燃料自带的烟量上，0 是几乎无烟。"]}
              actions={(
                <span className="be-panel-actions">
                  <button
                    type="button"
                    className="be-icon"
                    title="外观恢复默认"
                    aria-label="外观恢复默认"
                    onClick={() => setSim({ vorticity: simDefaults.vorticity, burning_rate: simDefaults.burning_rate, flame_temp: simDefaults.flame_temp, smoke: simDefaults.smoke })}
                  ><RotateCcw size={13} /></button>
                </span>
              )}
            >
              <SliderField label="涡度" value={sim.vorticity} range={limits.ranges?.vorticity} step={0.01} onCommit={(value) => setSim({ vorticity: value })} />
              <SliderField label="燃烧速度" value={sim.burning_rate} range={limits.ranges?.burning_rate} step={0.05} onCommit={(value) => setSim({ burning_rate: value })} />
              <SliderField label="火焰温度" value={sim.flame_temp} range={limits.ranges?.flame_temp} step={0.1} onCommit={(value) => setSim({ flame_temp: value })} />
              <SliderField label="烟量" value={sim.smoke} range={limits.ranges?.smoke} step={0.05} onCommit={(value) => setSim({ smoke: value })} />
            </Panel>
          )}
          {gas && (
            <Panel title="渲染" hint="Cycles 每个像素的采样数。多了噪点少，只影响出图，不影响模拟。">
              <SliderField label="采样" value={sim.samples} range={limits.ranges?.samples} step={1} onCommit={(value) => setSim({ samples: Math.round(value) })} />
            </Panel>
          )}
        </>
      )}
      {cameraPose && selected?.kind === "camera" && (
        <Panel
          title="摄像机"
          hint={["移动改相机位置，目标点保持不动。点视口里的目标点再拖，改的是它看向哪里。", "焦距 12–120 mm。转向工具不转相机。"]}
        >
          <div className="panel-actions">
            <button type="button" className="secondary-button" onClick={() => enterView(selected.viewpoint)}><Video size={13} />进入视角</button>
            {cameraOverrides?.[selected.viewpoint] && (
              <button type="button" className="secondary-button" onClick={() => {
                const map = { ...(cameraOverrides || {}) };
                delete map[selected.viewpoint];
                onCameras(map);
              }}><RotateCcw size={13} />恢复自动</button>
            )}
          </div>
          <VecFields title="位置" labels={["X", "Y", "Z"]} value={cameraPose.location} onCommit={(point) => commitCamera(selected.viewpoint, { location: point })} />
          <VecFields title="目标" labels={["X", "Y", "Z"]} value={cameraPose.target} onCommit={(point) => commitCamera(selected.viewpoint, { target: point })} />
          <MeterField label="焦距 mm" value={cameraPose.focal_mm} onCommit={(value) => commitCamera(selected.viewpoint, { focal_mm: value })} />
          {selected.part === "target" && <p className="field-hint">正在改目标点。</p>}
          {cameraPose.blocked && <p className="stage-note">视线被挡住了，仍可以保存。</p>}
          {cameraPose.fallback && <p className="stage-note">这个视角在场景里放不下，借用了侧视机位。</p>}
        </Panel>
      )}
      {item && (
        <>
          <Panel
            title="物体"
            actions={(
              <span className="be-panel-actions">
                <button type="button" className="be-icon" title="复制" aria-label="复制" disabled={full} onClick={() => setNote(onDuplicate(selected.index) || "")}><Copy size={13} /></button>
                <button type="button" className="be-icon" title="移出 (⌫)" aria-label="移出" onClick={() => removePiece(selected.index)}><Trash2 size={13} /></button>
              </span>
            )}
          >
            <label className="field">
              <span className="field-label">名称</span>
              <input maxLength={80} value={item.name || ""} onChange={(event) => onCommit(selected.index, { name: event.target.value })} />
            </label>
            <div className="field">
              <span className="field-label">尺寸</span>
              <span className="readonly-value">{Array.isArray(item.size) && item.size.length === 3 ? `${formatMeter(item.size[0])} × ${formatMeter(item.size[1])} × ${formatMeter(item.size[2])} m` : "未知"}</span>
            </div>
            {itemIssues.map((code) => <p className="stage-note" key={code}>{issueNote(code, domainCap)}</p>)}
          </Panel>
          <Panel title="变换" hint="求解器烧这个包围盒。朝向的单位是度，Z 是中心高度。">
            <VecFields title="位置" labels={["X", "Y", "Z"]} value={[item.x, item.y, item.z]} onCommit={(point) => onCommit(selected.index, { x: point[0], y: point[1], z: point[2] })} />
            <MeterField label="朝向 °" value={yawToDegrees(item.yaw || 0)} onCommit={(value) => {
              const yaw = degreesToYaw(value);
              if (yaw == null) return false;
              onCommit(selected.index, { yaw });
              return true;
            }} />
          </Panel>
          <Panel title="燃烧" hint="勾选后最近的一侧从第一帧就烧，其余逐段点着。没勾则等火烧过来。">
            <FuelSelect label="材料" value={item.fuel_type} fuels={fuels} onChange={(value) => onCommit(selected.index, { fuel_type: value })} />
            <label className="be-check">
              <input type="checkbox" checked={Boolean(item.ignition)} onChange={(event) => onCommit(selected.index, { ignition: event.target.checked })} />
              <span>一开始就烧</span>
            </label>
          </Panel>
        </>
      )}
      {pickedLight && (
        <>
          <Panel
            title="灯光"
            hint={[
              pickedLight.type === "SUN" ? "晴用这个强度，雨关掉，雪大约两成。顶盖盖住时自动太阳是 0。" : "",
              lightsAutomatic ? "这是自动灯。一改就会收成这个场景自己的灯，烘焙不再用天气里的那盏太阳。" : "",
            ]}
            actions={(
              <span className="be-panel-actions">
                <button type="button" className="be-icon" title="删除 (⌫)" aria-label="删除" onClick={() => removeLight(pickedLight.id)}><Trash2 size={13} /></button>
              </span>
            )}
          >
            <label className="field">
              <span className="field-label">名称</span>
              <input maxLength={80} value={pickedLight.name || ""} onChange={(event) => updateLight(pickedLight.id, { name: event.target.value })} />
            </label>
            <label className="field">
              <span className="field-label">类型</span>
              <select value={pickedLight.type} onChange={(event) => updateLight(pickedLight.id, { type: event.target.value })}>
                {Object.entries(LIGHT_LABEL).map(([type, label]) => <option key={type} value={type}>{label}</option>)}
              </select>
            </label>
            <label className="field color-field">
              <span className="field-label">颜色</span>
              <input type="color" value={rgbToHex(pickedLight.color)} onChange={(event) => updateLight(pickedLight.id, { color: hexToRgb(event.target.value) })} />
            </label>
            <MeterField label={pickedLight.type === "SUN" ? "晴天强度" : "强度 W"} value={pickedLight.energy} onCommit={(value) => updateLight(pickedLight.id, { energy: Math.max(0, value) })} />
            <label className="be-check">
              <input type="checkbox" checked={Boolean(pickedLight.enabled)} onChange={(event) => updateLight(pickedLight.id, { enabled: event.target.checked })} />
              <span>启用</span>
            </label>
          </Panel>
          <Panel title="变换" hint={pickedLight.type === "POINT" ? "点光没有朝向。转向工具只转太阳、聚光和面光。" : ""}>
            <VecFields title="位置" labels={["X", "Y", "Z"]} value={pickedLight.location} onCommit={(point) => updateLight(pickedLight.id, { location: clampLightPoint(point, site) })} />
            {pickedLight.type !== "POINT" && (
              <VecFields
                title="旋转 °"
                labels={["X", "Y", "Z"]}
                value={(pickedLight.rotation || [0, 0, 0]).map((turn) => yawToDegrees(turn || 0))}
                onCommit={(point) => {
                  const rotation = point.map((value) => degreesToYaw(value));
                  if (rotation.some((turn) => turn == null)) return false;
                  updateLight(pickedLight.id, { rotation });
                  return true;
                }}
              />
            )}
          </Panel>
          <Panel title="形状">
            {pickedLight.type === "SPOT" && (
              <MeterField label="光斑 °" value={yawToDegrees(pickedLight.spot_size || 0.8)} onCommit={(value) => {
                const turn = degreesToYaw(value);
                if (turn == null) return false;
                updateLight(pickedLight.id, { spot_size: Math.min(Math.PI, Math.max(0.05, turn)) });
                return true;
              }} />
            )}
            {pickedLight.type === "AREA" ? (
              <>
                <MeterField label="宽 m" value={pickedLight.size} onCommit={(value) => updateLight(pickedLight.id, { size: value })} />
                <MeterField label="高 m" value={pickedLight.size_y} onCommit={(value) => updateLight(pickedLight.id, { size_y: value })} />
              </>
            ) : (
              <MeterField label={pickedLight.type === "SUN" ? "太阳角" : "软阴影 m"} value={pickedLight.size} onCommit={(value) => updateLight(pickedLight.id, { size: value })} />
            )}
          </Panel>
        </>
      )}
      {floorInfo && (
        <Panel title="地板" hint="地板是地面，跟着场地的长宽走，不是一块可以挪的方块。">
          <div className="field">
            <span className="field-label">范围</span>
            <span className="readonly-value">{floorInfo.size.slice(0, 2).map((value) => formatMeter(value)).join(" × ")} m</span>
          </div>
        </Panel>
      )}
      {shellBox && (
        <Panel
          title="结构"
          hint="可以移动、缩放、改名和删除。火柱会在显示和烘焙里挖开实体，手柄拖的是完整方块。不燃烧的方块顶到场地高度时可以当顶盖，挡住雨雪。可燃和起火面不挡雨，起火面也不挪动起火点。每边至少 0.05 m，不能超出场地。"
          actions={(
            <span className="be-panel-actions">
              <button type="button" className="be-icon" title="删除 (⌫)" aria-label="删除" onClick={() => removeStructure(shellBox.id)}><Trash2 size={13} /></button>
            </span>
          )}
        >
          <label className="field">
            <span className="field-label">名称</span>
            <input maxLength={40} value={shellBox.name || ""} onChange={(event) => updateStructure(shellBox.id, { name: event.target.value })} />
          </label>
          <label className="field">
            <span className="field-label">角色</span>
            <select value={shellBox.role || "inert"} onChange={(event) => updateStructure(shellBox.id, { role: event.target.value })}>
              {STRUCTURE_ROLES.map((role) => <option key={role} value={role}>{KIND_LABEL[role]}</option>)}
            </select>
          </label>
          <VecFields title="中心" labels={["X", "Y", "Z"]} value={shellBox.center} onCommit={(point) => updateStructure(shellBox.id, { center: point })} />
          <VecFields title="尺寸 m" labels={["长", "宽", "高"]} value={shellBox.size} onCommit={(point) => updateStructure(shellBox.id, { size: point })} />
          <p className="field-hint">选中后按 G 拖动，按 S 拉伸。</p>
        </Panel>
      )}
      {!selected && <div className="be-empty">在大纲或视口里点一件，这里显示它的属性。</div>}
    </>
  );

  const sceneProps = (
    <>
      <Panel
        title="预设"
        hint={["套用会换成那一套结构、场地和默认摆放，并清掉起火点、摄像机、气体域和灯光。", "恢复默认摆放只恢复家具，不动结构、场地和灯。"]}
        actions={(
          <span className="be-panel-actions">
            <button
              type="button"
              className={confirmRestore ? "be-icon needed" : "be-icon"}
              title={confirmRestore ? "再点一次，恢复默认" : "恢复默认摆放"}
              aria-label={confirmRestore ? "再点一次，恢复默认" : "恢复默认摆放"}
              onClick={restore}
            ><RotateCcw size={13} /></button>
          </span>
        )}
      >
        <div className="field"><span className="field-label">当前</span><span className="readonly-value">{scenario}</span></div>
        <div className="be-seg wide" role="group" aria-label="预设">
          {(scenarios || []).map((name) => (
            <button type="button" key={name} className={confirmPreset === name || (scenario === name && !confirmPreset) ? "active" : ""} onClick={() => applyPreset(name)}>{name}</button>
          ))}
        </div>
        {confirmPreset && <p className="stage-note">再点一次「{confirmPreset}」，换成这套预设。结构和场地的改动会丢掉。</p>}
        {confirmRestore && <p className="stage-note">再点一次，恢复默认摆放</p>}
        {site && <VecFields title="场地 m" labels={["长", "宽", "高"]} value={site} onCommit={commitDomain} />}
        {origin && <div className="field"><span className="field-label">起火点</span><span className="readonly-value">{formatMeter(origin[0])}, {formatMeter(origin[1])}</span></div>}
        <div className="field"><span className="field-label">已摆</span><span className="readonly-value">{items.length} / {maxPlacements}</span></div>
      </Panel>
      <Panel title="起火点">
        <FuelSelect label="材料" value={originFuel} fuels={fuels} onChange={onOriginFuel} />
      </Panel>
      <Panel
        title="视口"
        hint={[
          "实体：固定的工作室光，跟着视角转，只为看清摆放，不进烘焙。",
          "材质预览：加一层室内环境反射，可以勾上场景灯光一起看。",
          "渲染预览：只用烘焙用的世界光和灯。室内天然很暗，和烘焙一样，火焰本身不在视口里。",
        ]}
      >
        <div className="be-seg wide">
          {VIEWS.map(([id, label]) => <button type="button" key={id} onClick={() => liveShell && aimAt(cameraFor(liveShell, id))}>{label}</button>)}
        </div>
        <div className="field">
          <span className="field-label">着色</span>
          {shadingSwitch}
        </div>
        {shading === "material" && sceneLightsCheck}
      </Panel>
    </>
  );

  return (
    <div className="scene-editor-page" tabIndex={0} ref={rootRef} onKeyDown={onKey}>
      <header className="be-topbar">
        {header}
        <div className="be-topbar-spacer" />
        {headerEnd}
      </header>
      <div className="be-body" ref={bodyRef} style={{ "--side-w": `${sideWidth}px` }}>
        <div className="be-main">
          <div className="be-vp-header">
            <div className="be-seg" role="group" aria-label="视图">
              {VIEWS.map(([id, label]) => <button type="button" key={id} onClick={() => liveShell && aimAt(cameraFor(liveShell, id))}>{label}</button>)}
            </div>
            {shadingSwitch}
            {shading === "material" && sceneLightsCheck}
            <span className="be-vp-info">
              {items.length} / {maxPlacements} 件
              {rejected ? <em><TriangleAlert size={12} />{rejected} 件提交会被拒绝</em> : null}
            </span>
            <div className="be-menu" ref={lightMenu}>
              <button
                type="button"
                className="be-btn"
                title={lightsAutomatic && !lights.length ? "灯光还在计算" : lights.length >= lightLimit ? `最多 ${lightLimit} 盏` : "添加光源"}
                disabled={lightBlocked}
                onClick={() => setAddingLight((open) => !open)}
              ><Plus size={13} />光源<ChevronDown size={12} /></button>
              {addingLight && (
                <div className="be-menu-pop" role="menu">
                  {Object.entries(LIGHT_LABEL).map(([type, label]) => (
                    <button type="button" role="menuitem" key={type} onClick={() => addLight(type)}><Lightbulb size={13} />{label}</button>
                  ))}
                </div>
              )}
            </div>
          </div>
          <div className="be-viewport" onPointerDown={() => rootRef.current?.focus()}>
            <SceneViewport scenario={scenario} shell={liveShell} items={items} assets={assets} selected={selected} tool={tool} aim={aim} ghost={ghost} cameras={cameras} gas={gas} origin={origin} lights={lights} shading={shading} sceneLights={Boolean(sceneLights)} world={preview?.world || null} hidden={hidden} editingId={editingId} onHover={hoverGhost} onDrop={dropGhost} onLook={(pose) => { const shot = bakeShot(shell, pose); if (shot) setAim({ ...shot, fov: focalFov(pose.focal_mm), token: Date.now() }); }} onSelect={select} onCommit={onCommit} onCommitOrigin={commitOrigin} onCommitCamera={commitCamera} onCommitGas={commitGas} onCommitLight={updateLight} onCommitStructure={updateStructure} onCanvas={onCanvas} />
            <div className="be-toolshelf" role="toolbar" aria-label="场景工具">
              <button type="button" className={tool === "translate" ? "be-tool active" : "be-tool"} title="移动 (G)" aria-label="移动" onClick={() => onTool("translate")}><Move3d size={16} /></button>
              <button type="button" className={tool === "rotate" ? "be-tool active" : "be-tool"} title="转向 (R)" aria-label="转向" onClick={() => onTool("rotate")}><Rotate3d size={16} /></button>
              <button type="button" className={tool === "scale" ? "be-tool active" : "be-tool"} title="缩放 (S)" aria-label="缩放" disabled={!scalable(selected)} onClick={() => onTool("scale")}><Scaling size={16} /></button>
              <span className="be-tool-sep" />
              <button type="button" className="be-tool" title="对准选中 (F)" aria-label="对准选中" onClick={frameSelection}><Focus size={16} /></button>
              <button type="button" className="be-tool" title="依次进入各视角。单击相机只选中" aria-label="看渲染相机" disabled={!cameras.length} onClick={lookThrough}><Video size={16} /></button>
            </div>
          </div>
          <section className={shelfOpen ? "be-asset-shelf open" : "be-asset-shelf"} aria-label="资产">
            <div className="be-shelf-head">
              <button type="button" className="be-ol-toggle" aria-expanded={shelfOpen} onClick={() => setShelfOpen((open) => !open)}>
                {shelfOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                <span>资产</span>
                <small>{(assets || []).length}</small>
              </button>
              <span className="be-shelf-hint">拖进视口放下，或单击自动找空位。求解器烧包围盒。最多 {maxPlacements} 件。</span>
              <a className="be-btn" href="#/library">上传 glb</a>
            </div>
            {shelfOpen && (
              <div className="be-shelf-pick">
                <div className="be-shelf-tools">
                  <input value={shelfQuery} placeholder="搜索模型" aria-label="搜索模型" onChange={(event) => setShelfQuery(event.target.value)} />
                  <div className="be-shelf-groups" role="group" aria-label="模型分组">
                    {shelfGroups.map((name) => <button type="button" key={name} className={shelfGroup === name ? "on" : ""} onClick={() => setShelfGroup(name)}>{name}</button>)}
                  </div>
                </div>
                <div className="be-shelf-grid">
                {shelfAssets.map((asset) => (
                  <button
                    key={asset.asset_id}
                    type="button"
                    className="be-asset"
                    disabled={full}
                    draggable={!full}
                    title={`${asset.label} · ${asset.fuel_type} · ${formatMeter(asset.size?.[0])}×${formatMeter(asset.size?.[1])}×${formatMeter(asset.size?.[2])} m${asset.url ? "" : " · 盒子"}`}
                    onDragStart={(event) => {
                      suppressClick.current = true;
                      dragAsset.current = asset;
                      event.dataTransfer.setData("application/x-firephantom-asset", asset.asset_id);
                      event.dataTransfer.setData("text/plain", asset.asset_id);
                      event.dataTransfer.effectAllowed = "copy";
                      const image = event.currentTarget.querySelector("img");
                      if (image) event.dataTransfer.setDragImage(image, 48, 48);
                    }}
                    onDragEnd={() => {
                      dragAsset.current = null;
                      window.setTimeout(() => { suppressClick.current = false; }, 0);
                    }}
                    onClick={() => {
                      if (suppressClick.current) {
                        suppressClick.current = false;
                        return;
                      }
                      add(asset);
                    }}
                  >
                    <AssetThumb asset={asset} className="be-asset-thumb" />
                    <span>{asset.label}</span>
                    <small>{asset.fuel_type}</small>
                  </button>
                ))}
                {!shelfAssets.length && <p className="be-ol-empty">没有匹配的模型。</p>}
                </div>
              </div>
            )}
          </section>
        </div>
        <div className="be-split-v" role="separator" aria-orientation="vertical" aria-label="调整侧栏宽度" onPointerDown={dragSide} />
        <aside className="be-side" ref={sideRef} style={{ gridTemplateRows: `minmax(80px, ${outlinerShare}fr) 5px minmax(120px, ${1 - outlinerShare}fr)` }}>
          <section className="be-outliner" aria-label="大纲">
            <div className="be-area-head"><span>大纲</span></div>
            <div className="be-outliner-scroll">
              <OutlinerGroup
                title="灯光"
                icon={Lightbulb}
                count={lights.length}
                open={isOpen("lights")}
                onToggle={() => flip("lights")}
              >
                {lights.map((light) => (
                  <OutlinerRow
                    key={light.id}
                    icon={Lightbulb}
                    label={light.name || LIGHT_LABEL[light.type]}
                    tag={`${LIGHT_LABEL[light.type]}${lightsAutomatic ? " · 自动" : ""}`}
                    dim={!light.enabled}
                    visible={visible(`light:${light.id}`)}
                    onToggleVisible={toggleVisible(`light:${light.id}`)}
                    picked={selected?.kind === "light" && selected.id === light.id}
                    title={`${LIGHT_LABEL[light.type]} · ${formatMeter(light.energy)}${light.type === "SUN" ? "" : " W"}${light.enabled ? "" : " · 关闭"}`}
                    onClick={() => select({ kind: "light", id: light.id })}
                  />
                ))}
                {!lights.length && <p className="be-ol-empty">{lightsAutomatic ? "灯光还在计算。" : "没有灯。烘焙里只剩环境光和火焰。"}</p>}
              </OutlinerGroup>
              <OutlinerGroup title="摄像机" icon={Video} count={cameras.length} open={isOpen("cameras")} onToggle={() => flip("cameras")}>
                {cameras.map((entry) => (
                  <OutlinerRow
                    key={entry.viewpoint}
                    icon={Video}
                    label={entry.viewpoint}
                    tag={cameraOverrides?.[entry.viewpoint] ? "自定义" : "自动"}
                    warn={entry.pose?.blocked || entry.pose?.fallback}
                    visible={visible(`camera:${entry.viewpoint}`)}
                    onToggleVisible={toggleVisible(`camera:${entry.viewpoint}`)}
                    picked={selected?.kind === "camera" && selected.viewpoint === entry.viewpoint}
                    title={`双击进入视角${entry.pose?.blocked ? " · 视线被挡" : ""}${entry.pose?.fallback ? " · 借用侧视" : ""}`}
                    onClick={() => select({ kind: "camera", viewpoint: entry.viewpoint, part: "body" })}
                    onDoubleClick={() => enterView(entry.viewpoint)}
                  />
                ))}
              </OutlinerGroup>
              <OutlinerGroup title="求解" icon={Flame} count={2} open={isOpen("solver")} onToggle={() => flip("solver")}>
                <OutlinerRow
                  icon={Flame}
                  label="起火点"
                  tag={originOverride ? "自定义" : "自动"}
                  visible={visible("origin")}
                  onToggleVisible={toggleVisible("origin")}
                  picked={selected?.kind === "origin"}
                  title={origin ? `${formatMeter(origin[0])}, ${formatMeter(origin[1])}, ${formatMeter(origin[2])}` : ""}
                  onClick={() => select({ kind: "origin" })}
                />
                <OutlinerRow
                  icon={Box}
                  label="气体域"
                  tag={gas ? (gasOverride ? "自定义" : "自动") : "计算中"}
                  visible={visible("domain")}
                  onToggleVisible={toggleVisible("domain")}
                  picked={selected?.kind === "domain"}
                  title={gas ? `${formatMeter(gas.scale[0] * 2)} × ${formatMeter(gas.scale[1] * 2)} × ${formatMeter(gas.scale[2] * 2)} m` : ""}
                  onClick={() => select({ kind: "domain" })}
                />
              </OutlinerGroup>
              <OutlinerGroup title="物体" icon={Cuboid} count={`${items.length}/${maxPlacements}`} open={isOpen("items")} onToggle={() => flip("items")}>
                {items.map((entry, index) => {
                  const issues = pieceIssues(entry, items, liveShell, domainCap, gasBounds);
                  return (
                    <OutlinerRow
                      key={`${entry.asset_id}-${index}`}
                      thumb={<AssetThumb asset={assetRecord(assets, entry)} className="be-ol-thumb" />}
                      label={entry.name || entry.asset_id}
                      tag={`${entry.fuel_type}${entry.ignition ? " · 起火" : ""}`}
                      warn={issues.length > 0}
                      visible={visible(`piece:${index}`)}
                      onToggleVisible={toggleVisible(`piece:${index}`)}
                      picked={selected?.kind === "piece" && selected.index === index}
                      title={issues.map((code) => ISSUE_LABEL[code]).join("、")}
                      onClick={() => select({ kind: "piece", index })}
                    />
                  );
                })}
                {!items.length && <p className="be-ol-empty">从下方资产架拖一件进视口，或单击。</p>}
              </OutlinerGroup>
              <OutlinerGroup
                title="结构"
                icon={Square}
                count={`${boxes.length}/${structureLimit}`}
                open={isOpen("shell")}
                onToggle={() => flip("shell")}
                action={(() => {
                  const anyShown = hidden["shell:floor"] !== true || boxes.some((box) => shellEyeOn(boxToObstacle(box), hidden));
                  return (
                    <span className="be-ol-actions">
                      <button
                        type="button"
                        className="be-ol-eye"
                        title={boxes.length >= structureLimit ? `最多 ${structureLimit} 块结构` : "添加结构"}
                        aria-label="添加结构"
                        disabled={!site || boxes.length >= structureLimit}
                        onClick={addStructure}
                      >
                        <Plus size={12} />
                      </button>
                      <button
                        type="button"
                        className={anyShown ? "be-ol-eye" : "be-ol-eye off"}
                        title={anyShown ? "隐藏全部结构" : "显示全部结构"}
                        aria-label={anyShown ? "隐藏全部结构" : "显示全部结构"}
                        onClick={() => setShellVisible(!anyShown)}
                      >
                        {anyShown ? <Eye size={12} /> : <EyeOff size={12} />}
                      </button>
                    </span>
                  );
                })()}
              >
                <OutlinerRow
                  icon={Square}
                  label="地板"
                  tag="不燃烧"
                  visible={hidden["shell:floor"] !== true}
                  onToggleVisible={toggleVisible("shell:floor")}
                  picked={selected?.kind === "shell" && selected.id === "floor"}
                  onClick={() => select({ kind: "shell", id: "floor" })}
                />
                {boxes.map((box) => {
                  const obstacle = boxToObstacle(box);
                  return (
                    <OutlinerRow
                      key={box.id}
                      icon={Cuboid}
                      label={box.name || box.id}
                      tag={KIND_LABEL[box.role] || "不燃烧"}
                      visible={shellEyeOn(obstacle, hidden)}
                      onToggleVisible={toggleShell(obstacle)}
                      picked={selected?.kind === "shell" && selected.id === box.id}
                      onClick={() => select({ kind: "shell", id: box.id })}
                    />
                  );
                })}
                {!boxes.length && <p className="be-ol-empty">没有结构方块。地板和场地还在。</p>}
              </OutlinerGroup>
            </div>
          </section>
          <div className="be-split-h" role="separator" aria-orientation="horizontal" aria-label="调整大纲高度" onPointerDown={dragSplit} />
          <section className="be-props" aria-label="属性">
            <nav className="be-tabs" aria-label="属性标签">
              {tabList.map((entry) => {
                const Icon = entry.icon;
                return (
                  <button
                    type="button"
                    key={entry.id}
                    className={entry.id === currentTab.id ? "be-tab active" : "be-tab"}
                    title={entry.label}
                    aria-label={entry.label}
                    aria-pressed={entry.id === currentTab.id}
                    onClick={() => setTab(entry.id)}
                  >
                    <Icon size={15} />
                    {entry.badge && <i className="be-tab-dot" />}
                  </button>
                );
              })}
            </nav>
            <div className="be-props-body">
              <div className="be-area-head">
                <TabIcon size={13} />
                <span>{currentTab.label}</span>
                {currentTab.id === "object" && objectTitle && <strong>{objectTitle}</strong>}
              </div>
              <div className="be-props-scroll">
                {currentTab.id === "object" ? objectProps : currentTab.id === "scene" ? sceneProps : currentTab.content}
              </div>
            </div>
          </section>
        </aside>
      </div>
      <footer className="be-statusbar">
        <button type="button" className={note ? "be-status-note warn" : "be-status-note"} title={note ? "点一下清除" : ""} onClick={() => setNote("")}>
          {note ? <><TriangleAlert size={12} />{note}</> : (objectTitle ? `选中：${objectTitle}` : "未选中")}
        </button>
        <span className="be-status-keys">{KEY_HINT}</span>
        <span className="be-status-count">{items.length}/{maxPlacements} 件 · {lights.length} 灯 · {cameras.length} 机</span>
      </footer>
    </div>
  );
}
