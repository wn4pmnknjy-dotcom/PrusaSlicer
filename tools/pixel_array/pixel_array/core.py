"""像素锁格摆放核心。

坐标约定：平台像素网格 bed[y, x]，y 向上增长，x 向右增长；格子 (gx, gy) 对应平台上
[x0 + gx*px, x0 + (gx+1)*px) × [y0 + gy*py, y0 + (gy+1)*py)。
每个零件先绕 Z 轴转到“最小占地”角度，外接框左下角放在像素边界上，平移量都是像素整数倍。
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
from scipy.spatial import ConvexHull, QhullError


@dataclass
class Machine:
    name: str
    res_x: int
    res_y: int
    width: float    # 屏幕宽 mm
    height: float   # 屏幕高 mm

    @property
    def px(self):
        return self.width / self.res_x

    @property
    def py(self):
        return self.height / self.res_y


MACHINES = [
    Machine("2K 光固化 (2560×1600, 0.035mm)", 2560, 1600, 89.6, 56.0),
    Machine("4K 光固化 (3840×2400, 0.0233mm)", 3840, 2400, 89.6, 56.0),
]


@dataclass
class Settings:
    machine: Machine = field(default_factory=lambda: MACHINES[0])
    spacing: float = 1.0        # 零件之间最小间距 mm
    margin: float = 1.0         # 平台边缘留白 mm
    min_footprint: bool = True  # 绕 Z 转到最小占地
    allow_rot90: bool = True    # 摆放时允许再转 90°
    drop_to_bed: bool = True    # 最低点落到 Z=0
    center_origin: bool = True  # 导出坐标以平台中心为原点
    nest: bool = False          # 复制件交错紧密排列（不排成规则行列）


@dataclass
class Part:
    name: str
    tris: np.ndarray            # (N, 3, 3) 原始三角形
    count: int = 1
    grid_x: int = 0             # 指定阵列列数，0 = 自动
    grid_y: int = 0             # 指定阵列行数，0 = 自动
    gap_x: float = 1.0          # 指定阵列时 X 间距 mm
    gap_y: float = 1.0          # 指定阵列时 Y 间距 mm

    @property
    def size(self):
        v = self.tris.reshape(-1, 3)
        return v.max(axis=0) - v.min(axis=0)

    def grid_shape(self):
        """返回 (列, 行)；不是指定阵列模式时返回 None。"""
        n = max(1, self.count)
        if self.grid_x > 0 and self.grid_y > 0:
            return self.grid_x, self.grid_y
        if self.grid_x > 0:
            return self.grid_x, -(-n // self.grid_x)
        if self.grid_y > 0:
            return -(-n // self.grid_y), self.grid_y
        return None

    def total(self):
        g = self.grid_shape()
        if g and self.grid_x > 0 and self.grid_y > 0:
            return g[0] * g[1]
        return max(0, self.count)


@dataclass
class Instance:
    part: int        # 零件序号
    angle: float     # 绕 Z 旋转角（度）
    gx: int          # 外接框左下角所在像素
    gy: int
    mask: np.ndarray = field(repr=False)   # 真实占地掩码 [y, x]


@dataclass
class Layout:
    settings: Settings
    instances: list
    unplaced: dict    # 零件序号 -> 放不下的数量

    def transforms(self, parts):
        """每个实例的 (零件序号, 4x4 变换)。"""
        s = self.settings
        m = s.machine
        x0 = -m.width / 2 if s.center_origin else 0.0
        y0 = -m.height / 2 if s.center_origin else 0.0
        out = []
        for inst in self.instances:
            R = rot_z(inst.angle)
            v = parts[inst.part].tris.reshape(-1, 3) @ R[:3, :3].T
            lo = v.min(axis=0)
            T = np.eye(4)
            T[0, 3] = x0 + inst.gx * m.px - lo[0]
            T[1, 3] = y0 + inst.gy * m.py - lo[1]
            T[2, 3] = -lo[2] if s.drop_to_bed else 0.0
            out.append((inst.part, T @ R))
        return out


def rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    R = np.eye(4)
    R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
    return R


def apply(tris, T):
    v = tris.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
    return v.reshape(tris.shape)


def min_footprint_angle(tris):
    """绕 Z 转多少度能让 XY 外接矩形面积最小；转完后 X 方向是长边。"""
    pts = np.unique(np.round(tris.reshape(-1, 3)[:, :2], 6), axis=0)
    try:
        hull = pts[ConvexHull(pts).vertices]
    except (QhullError, ValueError):
        return 0.0

    def extent(a):
        c, s = np.cos(a), np.sin(a)
        r = hull @ np.array([[c, s], [-s, c]])
        return r.max(axis=0) - r.min(axis=0)

    e0 = extent(0.0)
    best_a, best_area = 0.0, e0[0] * e0[1]
    edges = np.roll(hull, -1, axis=0) - hull
    for a in np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2)):
        e = extent(-a)
        if e[0] * e[1] < best_area * (1 - 1e-4):   # 没有明显变小就不转
            best_a, best_area = -a, e[0] * e[1]
    deg = np.degrees(best_a)
    e = extent(best_a)
    if e[1] > e[0] * (1 + 1e-6):
        deg += 90.0
    deg = float(np.mod(deg, 360.0))
    return 0.0 if min(deg, 360.0 - deg) < 1e-6 else deg


def rasterize(tris_xy, px, py):
    """把 XY 投影栅格化。tris_xy: (N,3,2)，已平移到外接框左下角为 (0,0)。

    像素中心落在三角形内、或含有顶点的像素都算占用，最后填洞。
    """
    u = tris_xy[..., 0] / px
    v = tris_xy[..., 1] / py
    nx = max(1, int(np.ceil(u.max() - 1e-6)))
    ny = max(1, int(np.ceil(v.max() - 1e-6)))
    mask = np.zeros((ny, nx), bool)

    vi = np.clip(np.floor(u).astype(int), 0, nx - 1).ravel()
    vj = np.clip(np.floor(v).astype(int), 0, ny - 1).ravel()
    mask[vj, vi] = True

    # 候选像素中心范围
    i0 = np.ceil(u.min(axis=1) - 0.5).astype(int)
    i1 = np.floor(u.max(axis=1) - 0.5).astype(int)
    j0 = np.ceil(v.min(axis=1) - 0.5).astype(int)
    j1 = np.floor(v.max(axis=1) - 0.5).astype(int)
    span = np.maximum(i1 - i0, j1 - j0) + 1
    ok = (i1 >= i0) & (j1 >= j0)
    if ok.any():
        bucket = np.ceil(np.log2(np.maximum(span, 1))).astype(int)
        for b in np.unique(bucket[ok]):
            idx = np.flatnonzero(ok & (bucket == b))
            s = 1 << int(b)
            off = np.arange(s)
            chunk = max(1, 4_000_000 // (s * s))
            for c in range(0, len(idx), chunk):
                k = idx[c:c + chunk]
                ci = i0[k, None, None] + off[None, None, :]
                cj = j0[k, None, None] + off[None, :, None]
                valid = (ci <= i1[k, None, None]) & (cj <= j1[k, None, None])
                qx, qy = ci + 0.5, cj + 0.5
                ax, ay = u[k, 0, None, None], v[k, 0, None, None]
                bx, by = u[k, 1, None, None], v[k, 1, None, None]
                cx, cy = u[k, 2, None, None], v[k, 2, None, None]
                d1 = (bx - ax) * (qy - ay) - (by - ay) * (qx - ax)
                d2 = (cx - bx) * (qy - by) - (cy - by) * (qx - bx)
                d3 = (ax - cx) * (qy - cy) - (ay - cy) * (qx - cx)
                eps = 1e-9
                inside = ((d1 >= -eps) & (d2 >= -eps) & (d3 >= -eps)) | \
                         ((d1 <= eps) & (d2 <= eps) & (d3 <= eps))
                hit = valid & inside
                hi = np.broadcast_to(ci, hit.shape)[hit]
                hj = np.broadcast_to(cj, hit.shape)[hit]
                keep = (hi >= 0) & (hi < nx) & (hj >= 0) & (hj < ny)
                mask[hj[keep], hi[keep]] = True
    return ndimage.binary_fill_holes(mask)


def halo(mask, rx, ry, pad):
    """按椭圆（像素半径 rx, ry）膨胀，四周各补 pad 个像素。"""
    padded = np.pad(mask, pad)
    if rx <= 0 and ry <= 0:
        return padded
    y, x = np.ogrid[-pad:pad + 1, -pad:pad + 1]
    disk = (x / max(rx, 1e-9)) ** 2 + (y / max(ry, 1e-9)) ** 2 <= 1.0
    return ndimage.binary_dilation(padded, structure=disk)


@dataclass
class _Variant:
    angle: float
    mask: np.ndarray          # 真实占地
    width_mm: float           # 旋转后外接框尺寸
    height_mm: float


def footprint(tris, angle, m):
    t = apply(tris, rot_z(angle))[..., :2]
    lo = t.reshape(-1, 2).min(axis=0)
    t = t - lo
    hi = t.reshape(-1, 2).max(axis=0)
    return _Variant(angle, rasterize(t, m.px, m.py), hi[0], hi[1])


class _Bed:
    def __init__(self, s):
        m = s.machine
        self.s = s
        self.rx = s.spacing / 2 / m.px + 1.0   # 多 1 像素抵消栅格误差，保证实际间距不小于设定值
        self.ry = s.spacing / 2 / m.py + 1.0
        self.P = int(np.ceil(max(self.rx, self.ry)))
        self.W, self.H = m.res_x, m.res_y
        self.mx = int(np.ceil(s.margin / m.px - 1e-9))
        self.my = int(np.ceil(s.margin / m.py - 1e-9))
        self.occ = np.zeros((self.H + 2 * self.P, self.W + 2 * self.P), np.float64)

    def dilate(self, mask):
        return halo(mask, self.rx, self.ry, self.P).astype(np.float64)

    def hit_map(self, mask, D):
        """hit[gy, gx] = 掩码左下角放在 (gx, gy) 时与已占用区域重叠的像素数；不可放处为 inf。"""
        h, w = mask.shape
        if h > self.H - 2 * self.my or w > self.W - 2 * self.mx:
            return None
        if self.occ.any():
            hit = np.rint(fftconvolve(self.occ, D[::-1, ::-1], mode="valid"))
        else:
            hit = np.zeros((self.H - h + 1, self.W - w + 1))
        hit[:self.my, :] = np.inf
        hit[:, :self.mx] = np.inf
        hit[self.H - h - self.my + 1:, :] = np.inf
        hit[:, self.W - w - self.mx + 1:] = np.inf
        return hit

    @staticmethod
    def first_free(hit):
        free = hit < 0.5
        rows = np.flatnonzero(free.any(axis=1))
        if len(rows) == 0:
            return None
        gy = int(rows[0])
        gx = int(np.argmax(free[gy]))
        return gx, gy

    def put(self, D, gx, gy):
        h, w = D.shape
        self.occ[gy:gy + h, gx:gx + w] += D


def _self_corr(D):
    return np.rint(fftconvolve(D, D[::-1, ::-1], mode="full"))


def _add_shifted(hit, C, gx, gy, Dshape):
    H, W = Dshape
    y0, y1 = max(0, gy - H + 1), min(hit.shape[0], gy + H)
    x0, x1 = max(0, gx - W + 1), min(hit.shape[1], gx + W)
    hit[y0:y1, x0:x1] += C[y0 - gy + H - 1:y1 - gy + H - 1, x0 - gx + W - 1:x1 - gx + W - 1]


def _pitch(var, m, gap_x, gap_y):
    """阵列节距（像素整数）：外接框 + 间距，向上取整到像素。"""
    sx = max(1, int(np.ceil((var.width_mm + gap_x) / m.px - 1e-6)))
    sy = max(1, int(np.ceil((var.height_mm + gap_y) / m.py - 1e-6)))
    return sx, sy


def _grid_block(var, m, cols, n, gap_x, gap_y):
    """把 n 个同一零件按 cols 列、指定间距拼成一个整体掩码；返回 (掩码, 各件偏移)。"""
    sx, sy = _pitch(var, m, gap_x, gap_y)
    rows = -(-n // cols)
    h, w = var.mask.shape
    block = np.zeros(((rows - 1) * sy + h, (min(n, cols) - 1) * sx + w), bool)
    offs = []
    for k in range(n):
        r, c = divmod(k, cols)
        ox, oy = c * sx, r * sy
        block[oy:oy + h, ox:ox + w] |= var.mask
        offs.append((ox, oy))
    return block, offs


def _auto_grid(var, m, bed, n, gap):
    """自动阵列：一行尽量多放（按平台宽度），行数不超过平台高度。"""
    sx, sy = _pitch(var, m, gap, gap)
    h, w = var.mask.shape
    aw, ah = bed.W - 2 * bed.mx, bed.H - 2 * bed.my
    if w > aw or h > ah:
        return None
    cols = min(n, (aw - w) // sx + 1)
    rows = min(-(-n // cols), (ah - h) // sy + 1)
    nb = min(n, cols * rows)
    if nb < 2:
        return None
    return _grid_block(var, m, cols, nb, gap, gap)


def _best_slot(bed, items):
    """在候选 (朝向, 掩码, 偏移) 里挑放下后最靠左下的。"""
    best = None
    for var, mask, offs in items:
        D = bed.dilate(mask)
        hit = bed.hit_map(mask, D)
        if hit is None:
            continue
        pos = bed.first_free(hit)
        if pos is None:
            continue
        h, w = mask.shape
        key = (pos[1] + h, pos[0] + w)
        if best is None or key < best[0]:
            best = (key, var, mask, offs, D, hit, pos)
    return best


def arrange(parts, settings, progress=None, cancel=None):
    """把所有零件（含复制件）锁格摆到平台上，返回 Layout。

    复制件默认排成规则阵列（节距 = 最小占地外接框 + 间距，取像素整数倍）；
    settings.nest 为 True 时改成逐个找最靠左下的空位（交错紧密排列）。
    """
    s = settings
    m = s.machine
    bed = _Bed(s)
    extras = (0.0, 90.0) if s.allow_rot90 else (0.0,)

    jobs = []
    for i, p in enumerate(parts):
        n = p.total()
        if n <= 0:
            continue
        base = min_footprint_angle(p.tris) if s.min_footprint else 0.0
        variants = [footprint(p.tris, base + e, m) for e in extras]
        grid = p.grid_shape()
        blocks = []
        if grid:
            cols, rows = grid
            # 用户指定了 X/Y，就保持这个方向，不再试 90°
            blocks.append((variants[0], *_grid_block(variants[0], m, cols, min(n, cols * rows),
                                                     p.gap_x, p.gap_y)))
        elif n > 1 and not s.nest:
            for v in variants:
                g = _auto_grid(v, m, bed, n, s.spacing)
                if g:
                    blocks.append((v, *g))
        singles = [(v, v.mask, [(0, 0)]) for v in variants]
        area = max(int(b[1].sum()) for b in blocks) if blocks else int(variants[0].mask.sum()) * n
        jobs.append((area, i, n, blocks, singles))
    jobs.sort(key=lambda j: -j[0])

    instances, unplaced = [], {}
    total = sum(j[2] for j in jobs)
    done = [0]

    def tick(k=1):
        done[0] += k
        if progress:
            progress(done[0], total)
        if cancel and cancel():
            raise InterruptedError

    for _, i, n, blocks, singles in jobs:
        left = n
        # 1) 整块阵列
        best = _best_slot(bed, blocks) if blocks else None
        if best:
            _, var, mask, offs, D, hit, (gx, gy) = best
            bed.put(D, gx, gy)
            for ox, oy in offs:
                instances.append(Instance(i, var.angle, gx + ox, gy + oy, var.mask))
            left -= len(offs)
            tick(len(offs))
            singles = [t for t in singles if t[0] is var]   # 剩下的保持同一朝向
        if left <= 0:
            continue
        # 2) 剩下的（或整块放不下时）逐个找空位，同一朝向，增量更新冲突图
        best = _best_slot(bed, singles)
        if best is None:
            unplaced[i] = left
            tick(left)
            continue
        _, var, mask, offs, D, hit, pos = best
        C = None
        while True:
            bed.put(D, *pos)
            instances.append(Instance(i, var.angle, pos[0], pos[1], var.mask))
            left -= 1
            tick()
            if left == 0:
                break
            if C is None:
                C = _self_corr(D)
            _add_shifted(hit, C, pos[0], pos[1], D.shape)
            pos = bed.first_free(hit)
            if pos is None:
                unplaced[i] = left
                tick(left)
                break
    return Layout(s, instances, unplaced)


def preview_labels(layout):
    """平台像素图：0 = 空，k = 第 k-1 个零件。"""
    m = layout.settings.machine
    img = np.zeros((m.res_y, m.res_x), np.int16)
    for inst in layout.instances:
        h, w = inst.mask.shape
        view = img[inst.gy:inst.gy + h, inst.gx:inst.gx + w]
        view[inst.mask] = inst.part + 1
    return img


def export_stl(path, parts, layout, progress=None):
    from .stlio import StlWriter
    tf = layout.transforms(parts)
    with StlWriter(path) as w:
        for k, (i, T) in enumerate(tf):
            w.write(apply(parts[i].tris, T))
            if progress:
                progress(k + 1, len(tf))
    return len(tf)
