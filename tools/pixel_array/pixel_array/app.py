"""中文界面：拖入 STL → 设置数量/阵列 → 自动锁格摆放 → 导出 STL。"""
import os
import sys
import traceback

import numpy as np
from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QMainWindow, QMessageBox, QProgressBar, QPushButton, QSpinBox,
                               QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from . import __version__
from .core import MACHINES, Machine, Part, Settings, arrange, export_stl, preview_labels
from .stlio import read_stl

PALETTE = np.array([
    (66, 133, 244), (234, 67, 53), (251, 188, 5), (52, 168, 83), (171, 71, 188),
    (0, 172, 193), (255, 112, 67), (158, 157, 36), (92, 107, 192), (240, 98, 146),
    (38, 166, 154), (141, 110, 99)], np.uint8)
BED_BG = (40, 44, 52)


def part_color(i):
    return QColor(*PALETTE[i % len(PALETTE)])


class Preview(QWidget):
    """平台俯视图（Y 向上）。"""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(420, 280)
        self.machine = MACHINES[0]
        self.image = None
        self._buf = None

    def set_layout(self, machine, layout):
        self.machine = machine
        if layout is None:
            self.image = None
        else:
            lab = preview_labels(layout)[::-1]
            colors = np.vstack([np.array([BED_BG], np.uint8), PALETTE])
            idx = np.where(lab > 0, (lab - 1) % len(PALETTE) + 1, 0)
            self._buf = np.ascontiguousarray(colors[idx])
            h, w, _ = self._buf.shape
            self.image = QImage(self._buf.data, w, h, 3 * w, QImage.Format_RGB888)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(250, 250, 250))
        m = self.machine
        W, H = self.width() - 20, self.height() - 40
        sc = min(W / m.width, H / m.height)
        bw, bh = m.width * sc, m.height * sc
        x0 = (self.width() - bw) / 2
        y0 = 10 + (H - bh) / 2
        if self.image is not None:
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)
            p.drawImage(int(x0), int(y0), self.image.scaled(
                int(bw), int(bh), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        else:
            p.fillRect(int(x0), int(y0), int(bw), int(bh), QColor(*BED_BG))
            p.setPen(QColor(200, 200, 200))
            p.drawText(int(x0), int(y0), int(bw), int(bh), Qt.AlignCenter,
                       "把 STL 文件拖到窗口里\n然后点“自动摆放”")
        p.setPen(QPen(QColor(90, 90, 90), 1))
        p.drawRect(int(x0), int(y0), int(bw), int(bh))
        p.setPen(QColor(60, 60, 60))
        p.drawText(int(x0), int(y0 + bh + 6), int(bw), 20, Qt.AlignCenter,
                   f"平台 {m.width:g} × {m.height:g} mm，{m.res_x} × {m.res_y} 像素，"
                   f"像素 {m.px:.5g} mm")
        p.end()


class ArrayDialog(QDialog):
    """零件阵列：总数 + X/Y 数量和间距（X、Y 都为 0 = 自动紧密排列）。"""

    def __init__(self, part, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"零件阵列 - {part.name}")
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.total = QSpinBox()
        self.total.setRange(1, 100000)
        self.total.setValue(max(1, part.count))
        form.addRow("阵列零件总数", self.total)
        lay.addLayout(form)

        box = QGroupBox("阵列摆放")
        g = QGridLayout(box)
        g.addWidget(QLabel("阵列数量"), 0, 1)
        g.addWidget(QLabel("间距"), 0, 2)
        self.nx, self.ny = QSpinBox(), QSpinBox()
        self.gx, self.gy = QDoubleSpinBox(), QDoubleSpinBox()
        for r, (name, n, gap, nv, gv) in enumerate(
                [("X", self.nx, self.gx, part.grid_x, part.gap_x),
                 ("Y", self.ny, self.gy, part.grid_y, part.gap_y)], start=1):
            n.setRange(0, 1000)
            n.setValue(nv)
            n.setSpecialValueText("自动")
            gap.setRange(0, 50)
            gap.setDecimals(3)
            gap.setSingleStep(0.1)
            gap.setValue(gv)
            g.addWidget(QLabel(name), r, 0)
            g.addWidget(n, r, 1)
            g.addWidget(gap, r, 2)
            g.addWidget(QLabel("mm"), r, 3)
            n.valueChanged.connect(self._sync)
        lay.addWidget(box)
        tip = QLabel("X、Y 都选“自动”时：按零件最小位置自动排成行列，一行尽量放满（推荐）。\n"
                     "填了 X 或 Y：按指定行列和间距排成整块，节距自动取像素整数倍。")
        tip.setStyleSheet("color:#666")
        lay.addWidget(tip)
        bb = QDialogButtonBox()
        bb.addButton("确认", QDialogButtonBox.AcceptRole)
        bb.addButton("关闭", QDialogButtonBox.RejectRole)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._sync()

    def _sync(self):
        both = self.nx.value() > 0 and self.ny.value() > 0
        if both:
            self.total.setValue(self.nx.value() * self.ny.value())
        self.total.setEnabled(not both)
        self.gx.setEnabled(self.nx.value() > 0 or self.ny.value() > 0)
        self.gy.setEnabled(self.gx.isEnabled())

    def apply_to(self, part):
        part.count = self.total.value()
        part.grid_x, part.grid_y = self.nx.value(), self.ny.value()
        part.gap_x, part.gap_y = self.gx.value(), self.gy.value()


class CustomMachineDialog(QDialog):
    def __init__(self, m, parent=None):
        super().__init__(parent)
        self.setWindowTitle("自定义机型")
        f = QFormLayout(self)
        self.rx, self.ry = QSpinBox(), QSpinBox()
        self.w, self.h = QDoubleSpinBox(), QDoubleSpinBox()
        for s, v in ((self.rx, m.res_x), (self.ry, m.res_y)):
            s.setRange(16, 20000)
            s.setValue(v)
        for s, v in ((self.w, m.width), (self.h, m.height)):
            s.setRange(1, 1000)
            s.setDecimals(3)
            s.setValue(v)
        f.addRow("分辨率 X（像素）", self.rx)
        f.addRow("分辨率 Y（像素）", self.ry)
        f.addRow("屏幕宽 X（mm）", self.w)
        f.addRow("屏幕高 Y（mm）", self.h)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("确认")
        bb.button(QDialogButtonBox.Cancel).setText("取消")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def machine(self):
        return Machine("自定义", self.rx.value(), self.ry.value(), self.w.value(), self.h.value())


class Worker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.finished_ok.emit(self.fn(lambda d, t: self.progress.emit(d, t)))
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    COLS = ["", "文件名", "尺寸 X×Y×Z (mm)", "数量", "阵列方式"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"STL 像素锁格阵列工具 v{__version__}")
        self.resize(1200, 720)
        self.setAcceptDrops(True)
        self.parts = []
        self.layout_result = None
        self.worker = None
        self.custom = Machine("自定义", 2560, 1600, 89.6, 56.0)
        self.cfg = QSettings("PixelArray", "PixelArray")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # 第一行：文件操作
        row = QHBoxLayout()
        b_add = QPushButton("添加 STL…")
        b_add.clicked.connect(self.add_dialog)
        b_arr = QPushButton("阵列设置…")
        b_arr.clicked.connect(self.edit_array)
        b_del = QPushButton("删除选中")
        b_del.clicked.connect(self.delete_selected)
        b_clr = QPushButton("清空")
        b_clr.clicked.connect(self.clear)
        for b in (b_add, b_arr, b_del, b_clr):
            row.addWidget(b)
        row.addStretch(1)
        root.addLayout(row)

        # 第二行：机型和参数
        row = QHBoxLayout()
        row.addWidget(QLabel("机型"))
        self.machine_box = QComboBox()
        for m in MACHINES:
            self.machine_box.addItem(m.name)
        self.machine_box.addItem("自定义…")
        self.machine_box.activated.connect(self.on_machine)
        row.addWidget(self.machine_box)
        row.addWidget(QLabel("零件间距"))
        self.spacing = self._mm_spin(1.0)
        row.addWidget(self.spacing)
        row.addWidget(QLabel("边缘留白"))
        self.margin = self._mm_spin(1.0)
        row.addWidget(self.margin)
        self.cb_minfp = QCheckBox("转到最小占地")
        self.cb_rot90 = QCheckBox("允许转 90°")
        self.cb_drop = QCheckBox("落到平台 Z=0")
        self.cb_center = QCheckBox("导出以平台中心为原点")
        self.cb_nest = QCheckBox("复制件交错紧密排")
        self.cb_nest.setToolTip("不勾：复制件排成整齐的行列阵列。\n勾上：每个复制件找最靠左下的空位，"
                                "圆形等零件能交错排得更密。")
        for cb in (self.cb_minfp, self.cb_rot90, self.cb_drop, self.cb_center, self.cb_nest):
            cb.setChecked(cb is not self.cb_nest)
            cb.toggled.connect(self.invalidate)
            row.addWidget(cb)
        row.addStretch(1)
        root.addLayout(row)

        # 中间：零件表 + 预览
        split = QSplitter()
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        for c in (0, 2, 3, 4):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.cellDoubleClicked.connect(lambda r, c: self.edit_array() if c != 3 else None)
        split.addWidget(self.table)
        self.preview = Preview()
        split.addWidget(self.preview)
        split.setSizes([480, 720])
        root.addWidget(split, 1)

        # 底部
        row = QHBoxLayout()
        self.status = QLabel("把 STL 文件拖进窗口，设好数量后点“自动摆放”。")
        row.addWidget(self.status, 1)
        self.bar = QProgressBar()
        self.bar.setMaximumWidth(220)
        self.bar.setVisible(False)
        row.addWidget(self.bar)
        self.b_run = QPushButton("自动摆放")
        self.b_run.setMinimumHeight(34)
        self.b_run.clicked.connect(self.run_arrange)
        self.b_exp = QPushButton("导出 STL…")
        self.b_exp.setMinimumHeight(34)
        self.b_exp.setEnabled(False)
        self.b_exp.clicked.connect(self.run_export)
        row.addWidget(self.b_run)
        row.addWidget(self.b_exp)
        root.addLayout(row)

        self.load_cfg()
        self.preview.set_layout(self.machine(), None)

    # ---- 设置 ----
    def _mm_spin(self, v):
        s = QDoubleSpinBox()
        s.setRange(0, 50)
        s.setDecimals(3)
        s.setSingleStep(0.1)
        s.setSuffix(" mm")
        s.setValue(v)
        s.valueChanged.connect(self.invalidate)
        return s

    def machine(self):
        i = self.machine_box.currentIndex()
        return MACHINES[i] if i < len(MACHINES) else self.custom

    def on_machine(self, i):
        if i == len(MACHINES):
            d = CustomMachineDialog(self.custom, self)
            if d.exec():
                self.custom = d.machine()
            self.machine_box.setItemText(i, f"自定义 ({self.custom.res_x}×{self.custom.res_y}, "
                                            f"{self.custom.px:.4g}mm)…")
        self.invalidate()

    def settings(self):
        return Settings(machine=self.machine(), spacing=self.spacing.value(),
                        margin=self.margin.value(), min_footprint=self.cb_minfp.isChecked(),
                        allow_rot90=self.cb_rot90.isChecked(), drop_to_bed=self.cb_drop.isChecked(),
                        center_origin=self.cb_center.isChecked(), nest=self.cb_nest.isChecked())

    def load_cfg(self):
        c = self.cfg
        try:
            self.custom = Machine("自定义", int(c.value("cres_x", 2560)), int(c.value("cres_y", 1600)),
                                  float(c.value("cw", 89.6)), float(c.value("ch", 56.0)))
            self.machine_box.setCurrentIndex(int(c.value("machine", 0)))
            self.spacing.setValue(float(c.value("spacing", 1.0)))
            self.margin.setValue(float(c.value("margin", 1.0)))
            for k, cb in (("minfp", self.cb_minfp), ("rot90", self.cb_rot90),
                          ("drop", self.cb_drop), ("center", self.cb_center)):
                cb.setChecked(str(c.value(k, "true")).lower() == "true")
            self.cb_nest.setChecked(str(c.value("nest", "false")).lower() == "true")
        except (TypeError, ValueError):
            pass
        if self.machine_box.currentIndex() == len(MACHINES):
            self.machine_box.setItemText(len(MACHINES), f"自定义 ({self.custom.res_x}×"
                                         f"{self.custom.res_y}, {self.custom.px:.4g}mm)…")

    def save_cfg(self):
        c = self.cfg
        c.setValue("machine", self.machine_box.currentIndex())
        c.setValue("spacing", self.spacing.value())
        c.setValue("margin", self.margin.value())
        c.setValue("cres_x", self.custom.res_x)
        c.setValue("cres_y", self.custom.res_y)
        c.setValue("cw", self.custom.width)
        c.setValue("ch", self.custom.height)
        for k, cb in (("minfp", self.cb_minfp), ("rot90", self.cb_rot90),
                      ("drop", self.cb_drop), ("center", self.cb_center), ("nest", self.cb_nest)):
            c.setValue(k, "true" if cb.isChecked() else "false")

    def closeEvent(self, e):
        self.save_cfg()
        super().closeEvent(e)

    # ---- 零件表 ----
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls()]
        self.add_files([p for p in paths if p.lower().endswith(".stl")])

    def add_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择 STL 文件", "", "STL 文件 (*.stl)")
        self.add_files(paths)

    def add_files(self, paths):
        if not paths:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        bad = []
        try:
            for p in paths:
                try:
                    tris = read_stl(p)
                    if len(tris) == 0:
                        raise ValueError("空文件")
                    self.parts.append(Part(os.path.basename(p), tris))
                except Exception as ex:
                    bad.append(f"{os.path.basename(p)}：{ex}")
        finally:
            QApplication.restoreOverrideCursor()
        self.refresh_table()
        self.invalidate()
        if bad:
            QMessageBox.warning(self, "读取失败", "\n".join(bad))

    def refresh_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self.table.setRowCount(len(self.parts))
        for r, p in enumerate(self.parts):
            sw = QTableWidgetItem("  ")
            sw.setBackground(part_color(r))
            self.table.setItem(r, 0, sw)
            self.table.setItem(r, 1, QTableWidgetItem(p.name))
            sx, sy, sz = p.size
            self.table.setItem(r, 2, QTableWidgetItem(f"{sx:.2f} × {sy:.2f} × {sz:.2f}"))
            spin = QSpinBox()
            spin.setRange(0, 100000)
            spin.setValue(p.total())
            spin.setEnabled(not (p.grid_x > 0 and p.grid_y > 0))
            spin.valueChanged.connect(lambda v, part=p: self.on_count(part, v))
            self.table.setCellWidget(r, 3, spin)
            g = p.grid_shape()
            txt = "自动阵列" if g is None else \
                f"{g[0]} 列 × {g[1]} 行，间距 {p.gap_x:g}/{p.gap_y:g} mm"
            self.table.setItem(r, 4, QTableWidgetItem(txt))
        self.table.blockSignals(False)

    def on_count(self, part, v):
        part.count = v
        g = part.grid_shape()
        if g is not None:
            self.table.item(self.parts.index(part), 4).setText(
                f"{g[0]} 列 × {g[1]} 行，间距 {part.gap_x:g}/{part.gap_y:g} mm")
        self.invalidate()

    def selected_rows(self):
        return sorted({i.row() for i in self.table.selectedIndexes()})

    def edit_array(self):
        rows = self.selected_rows()
        if not rows:
            QMessageBox.information(self, "提示", "先在列表里选中一个零件。")
            return
        part = self.parts[rows[0]]
        d = ArrayDialog(part, self)
        if d.exec():
            d.apply_to(part)
            self.refresh_table()
            self.invalidate()

    def delete_selected(self):
        for r in reversed(self.selected_rows()):
            del self.parts[r]
        self.refresh_table()
        self.invalidate()

    def clear(self):
        self.parts.clear()
        self.refresh_table()
        self.invalidate()

    def invalidate(self, *_):
        self.layout_result = None
        self.b_exp.setEnabled(False)
        self.preview.set_layout(self.machine(), None)

    # ---- 计算 ----
    def busy(self, on, text=""):
        for w in (self.b_run, self.b_exp, self.table):
            w.setEnabled(not on)
        self.bar.setVisible(on)
        self.bar.setRange(0, 0)
        if text:
            self.status.setText(text)

    def on_progress(self, d, t):
        self.bar.setRange(0, max(1, t))
        self.bar.setValue(d)

    def start(self, fn, ok, text):
        self.busy(True, text)
        self.worker = Worker(fn)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(ok)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_failed(self, msg):
        self.busy(False, "出错了。")
        self.b_exp.setEnabled(self.layout_result is not None)
        QMessageBox.critical(self, "出错", msg)

    def run_arrange(self):
        if not any(p.total() > 0 for p in self.parts):
            QMessageBox.information(self, "提示", "先把 STL 文件拖进窗口。")
            return
        s = self.settings()
        self.save_cfg()
        parts = list(self.parts)
        self.start(lambda prog: arrange(parts, s, progress=prog),
                   self.on_arranged, "正在按像素锁格摆放…")

    def on_arranged(self, lay):
        self.layout_result = lay
        self.busy(False)
        self.b_exp.setEnabled(bool(lay.instances))
        self.preview.set_layout(lay.settings.machine, lay)
        msg = f"已摆放 {len(lay.instances)} 个零件。"
        if lay.unplaced:
            miss = "，".join(f"{self.parts[i].name} 差 {n} 个" for i, n in lay.unplaced.items())
            msg += f" 平台放不下：{miss}"
            QMessageBox.warning(self, "放不下", "平台空间不够，以下零件没有全部放下：\n" +
                                "\n".join(f"{self.parts[i].name}：少 {n} 个"
                                          for i, n in lay.unplaced.items()))
        self.status.setText(msg)

    def run_export(self):
        if not self.layout_result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出 STL", "阵列.stl", "STL 文件 (*.stl)")
        if not path:
            return
        if not path.lower().endswith(".stl"):
            path += ".stl"
        parts, lay = list(self.parts), self.layout_result
        self.start(lambda prog: (path, export_stl(path, parts, lay, prog)),
                   self.on_exported, "正在导出 STL…")

    def on_exported(self, res):
        path, n = res
        self.busy(False, f"已导出 {n} 个零件到 {path}")
        self.b_exp.setEnabled(True)
        QMessageBox.information(self, "完成", f"已导出 {n} 个零件：\n{path}\n\n"
                                "导入切片软件时请关闭“自动居中/自动摆放”，否则会破坏像素锁格。")


def selftest():
    """打包后自检：建窗口、摆两个零件、导出，成功返回 0。"""
    import tempfile
    from .core import apply, rot_z
    from .stlio import write_stl
    app = QApplication.instance() or QApplication(sys.argv)
    w = MainWindow()
    v = np.array([[x, y, z] for x in (0, 5) for y in (0, 3) for z in (0, 2)], float)
    f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "box.stl")
        write_stl(src, apply(v[np.array(f)], rot_z(20)))
        w.add_files([src])
        w.parts[0].count = 20
        lay = arrange(w.parts, w.settings())
        w.on_arranged(lay)
        n = export_stl(os.path.join(d, "out.stl"), w.parts, lay)
        ok = n == 20 and not lay.unplaced and len(read_stl(os.path.join(d, "out.stl"))) == 240
    app.quit()
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1:
        w.add_files([p for p in sys.argv[1:] if p.lower().endswith(".stl")])
    sys.exit(app.exec())
