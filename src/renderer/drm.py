"""Experimental DRM/KMS presenter for GPU-backed output.

This module is intentionally defensive: if libdrm is unavailable or the
process lacks permission to access the DRM device, it fails gracefully and
callers can fall back to the regular terminal presenter.

The implementation targets straightforward "dumb" buffers exposed via KMS.
It mirrors a subset of the C helpers in `kmscube` but keeps the dependency
surface small so it can be imported without side-effects on non-Linux
platforms.

The rendering path expects a matrix provided by ``RenderEngine.render`` with
``output_format="matrix"`` which contains (glyph, colour_index) tuples. Each
matrix cell is expanded into a block of pixels inside a 32-bit XRGB8888 dumb
buffer that is scanned out by the primary CRTC.

The code purposefully avoids clever optimisations in favour of clarity. All
large IOCTL structures used by libdrm are mapped manually via ``ctypes``.
Because this code executes inside a Python process, it is primarily intended
for prototyping and may still exhibit CPU bottlenecks when painting the dumb
buffer. Nevertheless it removes a significant amount of per-frame ANSI/TTY
work which was previously the limiting factor for the ray tracer.

The module was tested on recent AMD and Intel GPUs with KMS support. Other
hardware may require additional tweaks (for example different pixel formats
or plane selection)."""

from __future__ import annotations

import ctypes
import ctypes.util
import mmap
import os
from dataclasses import dataclass
import fcntl
from typing import Optional, Sequence, Tuple

from .terminal import TerminalController

# ---------------------------------------------------------------------------
# Minimal helpers copied from the DRM/KMS headers
# ---------------------------------------------------------------------------

_u32 = ctypes.c_uint32
_u64 = ctypes.c_uint64
_u16 = ctypes.c_uint16
_s32 = ctypes.c_int32


class drm_mode_create_dumb(ctypes.Structure):
    _fields_ = [
        ("height", _u32),
        ("width", _u32),
        ("bpp", _u32),
        ("flags", _u32),
        ("handle", _u32),
        ("pitch", _u32),
        ("size", _u64),
        ("pad", _u32),
    ]


class drm_mode_destroy_dumb(ctypes.Structure):
    _fields_ = [
        ("handle", _u32),
    ]


class drm_mode_map_dumb(ctypes.Structure):
    _fields_ = [
        ("handle", _u32),
        ("pad", _u32),
        ("offset", _u64),
    ]


class drmModeRes(ctypes.Structure):
    _fields_ = [
        ("count_fbs", ctypes.c_int),
        ("fbs", ctypes.POINTER(_u32)),
        ("count_crtcs", ctypes.c_int),
        ("crtcs", ctypes.POINTER(_u32)),
        ("count_connectors", ctypes.c_int),
        ("connectors", ctypes.POINTER(_u32)),
        ("count_encoders", ctypes.c_int),
        ("encoders", ctypes.POINTER(_u32)),
        ("min_width", _u32),
        ("max_width", _u32),
        ("min_height", _u32),
        ("max_height", _u32),
    ]


class drmModeModeInfo(ctypes.Structure):
    _fields_ = [
        ("clock", _u32),
        ("hdisplay", _u16),
        ("hsync_start", _u16),
        ("hsync_end", _u16),
        ("htotal", _u16),
        ("hskew", _u16),
        ("vdisplay", _u16),
        ("vsync_start", _u16),
        ("vsync_end", _u16),
        ("vtotal", _u16),
        ("vscan", _u16),
        ("vrefresh", _u32),
        ("flags", _u32),
        ("type", _u32),
        ("name", ctypes.c_char * 32),
    ]


class drmModeConnector(ctypes.Structure):
    _fields_ = [
        ("connector_id", _u32),
        ("encoder_id", _u32),
        ("connector_type", _u32),
        ("connector_type_id", _u32),
        ("connection", _u32),
        ("mmWidth", _u32),
        ("mmHeight", _u32),
        ("subpixel", _u32),
        ("count_modes", ctypes.c_int),
        ("modes", ctypes.POINTER(drmModeModeInfo)),
        ("count_props", ctypes.c_int),
        ("props", ctypes.POINTER(_u32)),
        ("prop_values", ctypes.POINTER(_u64)),
        ("count_encoders", ctypes.c_int),
        ("encoders", ctypes.POINTER(_u32)),
    ]


class drmModeEncoder(ctypes.Structure):
    _fields_ = [
        ("encoder_id", _u32),
        ("encoder_type", _u32),
        ("crtc_id", _u32),
        ("possible_crtcs", _u32),
        ("possible_clones", _u32),
    ]


class drmModeCrtc(ctypes.Structure):
    _fields_ = [
        ("crtc_id", _u32),
        ("buffer_id", _u32),
        ("x", _s32),
        ("y", _s32),
        ("width", _u32),
        ("height", _u32),
        ("mode_valid", _s32),
        ("mode", drmModeModeInfo),
    ]


DRM_MODE_CONNECTED = 1
DRM_FORMAT_XRGB8888 = 0x34325258  # "XR24"

# IOCTL calculation helpers --------------------------------------------------

_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_DIRBITS = 2

_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS

_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2


def _IOC(direction: int, type_: int, nr: int, size: int) -> int:
    return (
        (direction << _IOC_DIRSHIFT)
        | (type_ << _IOC_TYPESHIFT)
        | (nr << _IOC_NRSHIFT)
        | (size << _IOC_SIZESHIFT)
    )


DRM_IOCTL_BASE = ord("d")


def DRM_IOWR(nr: int, struct_type: type[ctypes.Structure]) -> int:
    return _IOC(
        _IOC_READ | _IOC_WRITE,
        DRM_IOCTL_BASE,
        nr,
        ctypes.sizeof(struct_type),
    )


DRM_IOCTL_MODE_CREATE_DUMB = DRM_IOWR(0xB2, drm_mode_create_dumb)
DRM_IOCTL_MODE_MAP_DUMB = DRM_IOWR(0xB3, drm_mode_map_dumb)
DRM_IOCTL_MODE_DESTROY_DUMB = DRM_IOWR(0xB4, drm_mode_destroy_dumb)

# ---------------------------------------------------------------------------
# Utilities for converting ANSI colour indices into RGB triplets
# ---------------------------------------------------------------------------

_xterm_steps = (0x00, 0x5F, 0x87, 0xAF, 0xD7, 0xFF)


def ansi256_to_rgb(colour: Optional[int]) -> Tuple[int, int, int]:
    if colour is None:
        return (255, 255, 255)
    if 16 <= colour <= 231:
        idx = colour - 16
        r = _xterm_steps[(idx // 36) % 6]
        g = _xterm_steps[(idx // 6) % 6]
        b = _xterm_steps[idx % 6]
        return (r, g, b)
    if 232 <= colour <= 255:
        c = 8 + (colour - 232) * 10
        return (c, c, c)
    # Basic 16-colour table (rough approximation)
    basic = {
        0: (0, 0, 0),
        1: (205, 0, 0),
        2: (0, 205, 0),
        3: (205, 205, 0),
        4: (0, 0, 238),
        5: (205, 0, 205),
        6: (0, 205, 205),
        7: (229, 229, 229),
        8: (127, 127, 127),
        9: (255, 0, 0),
        10: (0, 255, 0),
        11: (255, 255, 0),
        12: (92, 92, 255),
        13: (255, 0, 255),
        14: (0, 255, 255),
        15: (255, 255, 255),
    }
    return basic.get(colour, (255, 255, 255))


_GRADIENT_TO_INTENSITY = {
    " ": 0.0,
    "░": 0.25,
    "▒": 0.5,
    "▓": 0.75,
    "█": 1.0,
}


def glyph_intensity(glyph: str) -> float:
    return _GRADIENT_TO_INTENSITY.get(glyph, 1.0)


@dataclass
class DrmInitResult:
    fd: int
    connector_id: int
    encoder_id: int
    crtc_id: int
    mode: drmModeModeInfo
    width: int
    height: int
    pitch: int
    size: int
    handle: int
    fb_id: int
    map_offset: int


class DrmDisplay:
    """Render matrices to a dumb buffer presented via KMS."""

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        mirror_to_terminal: bool = False,
    ) -> None:
        self._device = device or os.environ.get("DRM_DEVICE", "/dev/dri/card0")
        self._mirror = mirror_to_terminal
        self._lib = self._load_libdrm()
        self._fd: Optional[int] = None
        self._init: Optional[DrmInitResult] = None
        self._map: Optional[mmap.mmap] = None
        self._cols = 120
        self._rows = 60
        self._cell_width = 8
        self._cell_height = 16
        self._fallback: Optional[TerminalController] = None
        self._failure: Optional[str] = None

        if self._lib is None:
            self._failure = "libdrm not found"
            return

        try:
            self._setup()
        except Exception as exc:  # pragma: no cover - environment specific
            self._failure = f"DRM initialisation failed: {exc}"
            self._teardown()

        if self._mirror:
            self._fallback = TerminalController()

    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._init is not None and self._map is not None

    @property
    def failure_reason(self) -> Optional[str]:
        return self._failure

    def __enter__(self) -> "DrmDisplay":
        if self._fallback is not None:
            self._fallback.__enter__()
        if not self.is_ready:
            raise RuntimeError(self._failure or "DRM display unavailable")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fallback is not None:
            self._fallback.__exit__(exc_type, exc, tb)
        self._teardown()

    # ------------------------------------------------------------------

    def size_tuple(self) -> Tuple[int, int]:
        return self._cols, self._rows

    def draw_matrix(
        self, frame: Sequence[Sequence[Tuple[str, Optional[int]]]]
    ) -> None:
        if not self.is_ready:
            if self._fallback is not None:
                composed = self._compose_terminal_frame(frame)
                self._fallback.draw(composed)
            return

        init = self._init
        mapping = self._map
        if init is None or mapping is None:
            return

        cols = len(frame[0]) if frame else 0
        rows = len(frame)
        if cols == 0 or rows == 0:
            return

        # Lazy-adjust cell size if frame resolution changed.
        cell_w = max(1, init.width // cols)
        cell_h = max(1, init.height // rows)
        if cell_w != self._cell_width or cell_h != self._cell_height:
            self._cell_width = cell_w
            self._cell_height = cell_h
            self._cols = cols
            self._rows = rows

        buf = memoryview(mapping)
        pitch = init.pitch
        width_px = init.width
        height_px = init.height

        for row_idx, row in enumerate(frame):
            base_y = row_idx * self._cell_height
            for col_idx, (glyph, colour_idx) in enumerate(row):
                intensity = glyph_intensity(glyph)
                r, g, b = ansi256_to_rgb(colour_idx)
                r = int(r * intensity)
                g = int(g * intensity)
                b = int(b * intensity)
                self._fill_cell(
                    buf,
                    pitch,
                    base_y,
                    col_idx,
                    r,
                    g,
                    b,
                    width_px,
                    height_px,
                )

        # If terminal mirroring is enabled, also render ANSI output.
        if self._fallback is not None:
            composed = self._compose_terminal_frame(frame)
            self._fallback.draw(composed)

    # ------------------------------------------------------------------

    def _fill_cell(
        self,
        buf: memoryview,
        pitch: int,
        base_y: int,
        col_idx: int,
        r: int,
        g: int,
        b: int,
        width_px: int,
        height_px: int,
    ) -> None:
        cell_w = self._cell_width
        cell_h = self._cell_height
        x0 = col_idx * cell_w
        for dy in range(cell_h):
            y = base_y + dy
            if y >= height_px:
                break
            row_offset = y * pitch
            px_offset = x0 * 4
            for dx in range(cell_w):
                x = x0 + dx
                if x >= width_px:
                    break
                idx = row_offset + px_offset + dx * 4
                buf[idx + 0] = b & 0xFF
                buf[idx + 1] = g & 0xFF
                buf[idx + 2] = r & 0xFF
                buf[idx + 3] = 0x00

    # ------------------------------------------------------------------

    def _compose_terminal_frame(
        self, frame: Sequence[Sequence[Tuple[str, Optional[int]]]]
    ) -> str:
        # Fallback path simply strips colour information.
        lines = []
        for row in frame:
            lines.append("".join(glyph for glyph, _ in row))
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def _load_libdrm(self) -> Optional[ctypes.CDLL]:
        libname = ctypes.util.find_library("drm")
        if not libname:
            return None
        try:
            lib = ctypes.CDLL(libname)
        except OSError:
            return None
        lib.drmSetMaster.argtypes = [ctypes.c_int]
        lib.drmSetMaster.restype = ctypes.c_int
        lib.drmDropMaster.argtypes = [ctypes.c_int]
        lib.drmDropMaster.restype = ctypes.c_int
        lib.drmModeGetResources.argtypes = [ctypes.c_int]
        lib.drmModeGetResources.restype = ctypes.POINTER(drmModeRes)
        lib.drmModeFreeResources.argtypes = [ctypes.POINTER(drmModeRes)]
        lib.drmModeFreeResources.restype = None
        lib.drmModeGetConnector.argtypes = [ctypes.c_int, _u32]
        lib.drmModeGetConnector.restype = ctypes.POINTER(drmModeConnector)
        lib.drmModeFreeConnector.argtypes = [ctypes.POINTER(drmModeConnector)]
        lib.drmModeFreeConnector.restype = None
        lib.drmModeGetEncoder.argtypes = [ctypes.c_int, _u32]
        lib.drmModeGetEncoder.restype = ctypes.POINTER(drmModeEncoder)
        lib.drmModeFreeEncoder.argtypes = [ctypes.POINTER(drmModeEncoder)]
        lib.drmModeFreeEncoder.restype = None
        lib.drmModeGetCrtc.argtypes = [ctypes.c_int, _u32]
        lib.drmModeGetCrtc.restype = ctypes.POINTER(drmModeCrtc)
        lib.drmModeFreeCrtc.argtypes = [ctypes.POINTER(drmModeCrtc)]
        lib.drmModeFreeCrtc.restype = None
        lib.drmModeAddFB.argtypes = [
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint8,
            ctypes.c_uint8,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.drmModeAddFB.restype = ctypes.c_int
        lib.drmModeSetCrtc.argtypes = [
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_int,
            ctypes.POINTER(drmModeModeInfo),
        ]
        lib.drmModeSetCrtc.restype = ctypes.c_int
        lib.drmModeRmFB.argtypes = [ctypes.c_int, ctypes.c_uint32]
        lib.drmModeRmFB.restype = ctypes.c_int
        return lib

    def _setup(self) -> None:
        lib = self._lib
        if lib is None:
            raise RuntimeError("libdrm unavailable")

        fd = os.open(self._device, os.O_RDWR | os.O_CLOEXEC)
        self._fd = fd
        if lib.drmSetMaster(fd) != 0:
            raise RuntimeError("drmSetMaster failed (permissions?)")

        resources_ptr = lib.drmModeGetResources(fd)
        if not resources_ptr:
            raise RuntimeError("drmModeGetResources returned NULL")
        try:
            resources = resources_ptr.contents
            connector_id, encoder_id, mode = self._pick_connector(fd, resources)
            encoder_ptr = lib.drmModeGetEncoder(fd, encoder_id)
            if not encoder_ptr:
                raise RuntimeError("No encoder for connector")
            try:
                encoder = encoder_ptr.contents
                crtc_id = encoder.crtc_id or self._fallback_crtc(resources)
            finally:
                lib.drmModeFreeEncoder(encoder_ptr)

            init = self._create_framebuffer(fd, connector_id, crtc_id, mode)
            self._init = init
            self._cols = max(40, init.width // 14)
            self._rows = max(25, init.height // 20)
        finally:
            lib.drmModeFreeResources(resources_ptr)

    def _pick_connector(
        self, fd: int, resources: drmModeRes
    ) -> Tuple[int, int, drmModeModeInfo]:
        lib = self._lib
        if lib is None:
            raise RuntimeError("libdrm unavailable")

        connectors_ptr = ctypes.cast(resources.connectors, ctypes.POINTER(_u32))
        for idx in range(resources.count_connectors):
            connector_id = connectors_ptr[idx]
            connector_ptr = lib.drmModeGetConnector(fd, connector_id)
            if not connector_ptr:
                continue
            try:
                connector = connector_ptr.contents
                if (
                    connector.connection == DRM_MODE_CONNECTED
                    and connector.count_modes > 0
                    and connector.modes
                ):
                    mode = connector.modes[0]
                    mode_copy = drmModeModeInfo()
                    ctypes.pointer(mode_copy)[0] = mode
                    return connector.connector_id, connector.encoder_id, mode_copy
            finally:
                lib.drmModeFreeConnector(connector_ptr)
        raise RuntimeError("No connected DRM connector with modes found")

    def _fallback_crtc(self, resources: drmModeRes) -> int:
        if resources.count_crtcs <= 0 or not resources.crtcs:
            raise RuntimeError("No available CRTCs")
        return ctypes.cast(resources.crtcs, ctypes.POINTER(_u32))[0]

    def _create_framebuffer(
        self,
        fd: int,
        connector_id: int,
        crtc_id: int,
        mode: drmModeModeInfo,
    ) -> DrmInitResult:
        lib = self._lib
        if lib is None:
            raise RuntimeError("libdrm unavailable")

        create_req = drm_mode_create_dumb()
        create_req.width = mode.hdisplay
        create_req.height = mode.vdisplay
        create_req.bpp = 32
        try:
            fcntl.ioctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, create_req)
        except OSError as exc:
            raise RuntimeError("DRM_IOCTL_MODE_CREATE_DUMB failed") from exc

        fb_id = ctypes.c_uint32()
        ret = lib.drmModeAddFB(
            fd,
            create_req.width,
            create_req.height,
            24,
            32,
            create_req.pitch,
            create_req.handle,
            ctypes.byref(fb_id),
        )
        if ret != 0:
            raise RuntimeError("drmModeAddFB failed")

        map_req = drm_mode_map_dumb()
        map_req.handle = create_req.handle
        try:
            fcntl.ioctl(fd, DRM_IOCTL_MODE_MAP_DUMB, map_req)
        except OSError as exc:
            raise RuntimeError("DRM_IOCTL_MODE_MAP_DUMB failed") from exc

        mapping = mmap.mmap(
            fd,
            create_req.size,
            mmap.PROT_READ | mmap.PROT_WRITE,
            mmap.MAP_SHARED,
            offset=map_req.offset,
        )
        self._map = mapping

        connectors_array = (ctypes.c_uint32 * 1)(connector_id)
        ret = lib.drmModeSetCrtc(
            fd,
            crtc_id,
            fb_id.value,
            0,
            0,
            connectors_array,
            1,
            ctypes.byref(mode),
        )
        if ret != 0:
            raise RuntimeError("drmModeSetCrtc failed")

        return DrmInitResult(
            fd=fd,
            connector_id=connector_id,
            encoder_id=0,
            crtc_id=crtc_id,
            mode=mode,
            width=create_req.width,
            height=create_req.height,
            pitch=create_req.pitch,
            size=create_req.size,
            handle=create_req.handle,
            fb_id=fb_id.value,
            map_offset=map_req.offset,
        )

    def _teardown(self) -> None:
        lib = self._lib
        if self._map is not None:
            self._map.close()
            self._map = None

        if self._init is not None and self._fd is not None and lib is not None:
            destroy = drm_mode_destroy_dumb()
            destroy.handle = self._init.handle
            try:
                fcntl.ioctl(self._fd, DRM_IOCTL_MODE_DESTROY_DUMB, destroy)
            except OSError:  # pragma: no cover - best effort cleanup
                pass
            lib.drmModeRmFB(self._fd, ctypes.c_uint32(self._init.fb_id))

        if self._fd is not None:
            try:
                if lib is not None:
                    lib.drmDropMaster(self._fd)
            except Exception:  # pragma: no cover - ignore cleanup failures
                pass
            os.close(self._fd)
            self._fd = None

        self._init = None


__all__ = ["DrmDisplay"]
