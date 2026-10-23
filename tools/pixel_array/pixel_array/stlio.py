"""STL 读写（二进制 / ASCII），只依赖 numpy。三角形数组形状为 (N, 3, 3)。"""
import os
import struct

import numpy as np

_BIN_DTYPE = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])


def read_stl(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(84)
        if len(head) == 84:
            n = struct.unpack("<I", head[80:84])[0]
            if 84 + n * 50 == size:
                data = np.fromfile(f, dtype=_BIN_DTYPE, count=n)
                return data["v"].astype(np.float64)
    return _read_ascii(path)


def _read_ascii(path):
    verts = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("vertex"):
                parts = s.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not verts or len(verts) % 3:
        raise ValueError("不是有效的 STL 文件")
    return np.asarray(verts, np.float64).reshape(-1, 3, 3)


def _normals(tris):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return n / ln


class StlWriter:
    """流式写二进制 STL，最后回填三角形总数。"""

    def __init__(self, path):
        self.f = open(path, "wb")
        self.f.write(b"pixel_array".ljust(80, b" "))
        self.f.write(struct.pack("<I", 0))
        self.count = 0

    def write(self, tris):
        rec = np.zeros(len(tris), dtype=_BIN_DTYPE)
        rec["normal"] = _normals(tris)
        rec["v"] = tris
        rec.tofile(self.f)
        self.count += len(tris)

    def close(self):
        self.f.seek(80)
        self.f.write(struct.pack("<I", self.count))
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_stl(path, tris):
    with StlWriter(path) as w:
        w.write(tris)
