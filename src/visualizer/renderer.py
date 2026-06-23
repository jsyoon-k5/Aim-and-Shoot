"""
Episode Video Renderer

Renders AimandShootEpisodeRecorder episodes as video files using Panda3D for
3D visualization.  Visual conventions (background sphere, camera HPR mapping,
target sphere placement) are kept identical to engine.py.

Coordinate mapping
------------------
Game Cartesian  (x, y, z):
    x = cos(az)*cos(el)   — forward at az=0, el=0
    y = sin(el)           — up
    z = sin(az)*cos(el)   — right at az>0

Panda3D world   (X, Y, Z):
    Y forward, X right, Z up

Therefore:  panda_X = game_z,  panda_Y = game_x,  panda_Z = game_y
Camera HPR:  heading = -azim,  pitch = elev,  roll = 0   (identical to engine.py)

Code written by June-Seop Yoon
"""

import numpy as np
import cv2
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from panda3d.core import (
    AmbientLight,
    CardMaker,
    DirectionalLight,
    GraphicsOutput,
    LineSegs,
    PNMImage,
    TextNode,
    Texture,
    TransparencyAttrib,
    Vec3,
    loadPrcFileData,
)
from direct.showbase.ShowBase import ShowBase

from ..agent.task import AimandShootSpiderShotTask as AnSTask
from ..agent.simulator import AimandShootEpisodeRecorder
from ..configs.constants import FEATURES, TINTERVAL
from .render_primitives import draw_background_sphere, draw_solid_sphere, TEXTURE_PATHS


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_RENDER_CONFIG = dict(
    width=1920,
    height=1080,
    fps=60,
    freeze_ms=0,         # optional final-frame hold duration in ms
    output_dir=".",
    target_color=(1.0, 1.0, 1.0, 1.0),
    crosshair_color=(0.0, 1.0, 0.0, 1.0),
    crosshair_radius=0.012,
    crosshair_thickness=2.0,
    gaze_color=(1.0, 0.0, 0.0, 0.95),
    gaze_trail_color=(1.0, 0.0, 0.0, 0.45),
    gaze_marker_size=0.025,
    gaze_trail_ms=250.0,
    gaze_trail_thickness=2.5,
    panda_cache_dir=None,
)

_PANDA_INITIALIZED = False   # Guard: loadPrcFileData should run once per process


def _default_panda_cache_dir() -> Path:
    """Return a user-local Panda3D cache dir outside the cloned repository."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / "Aim-and-Shoot" / "Panda3D"
        user_profile = os.environ.get("USERPROFILE")
        if user_profile:
            return Path(user_profile) / "AppData" / "Local" / "Aim-and-Shoot" / "Panda3D"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "Aim-and-Shoot" / "Panda3D"

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
    return base / "aim-and-shoot" / "panda3d"


def _panda_prc_path(path: Path) -> str:
    return Path(path).expanduser().resolve().as_posix()


def _ensure_panda_cache_dir(path: Path) -> Path:
    candidates = [
        Path(path).expanduser(),
        Path(tempfile.gettempdir()) / "Aim-and-Shoot" / "Panda3D",
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue
    return candidates[-1]


def _game_to_panda(pos):
    """
    Convert a game Cartesian position (x, y, z) to Panda3D world (X, Y, Z).

        panda_X = game_z
        panda_Y = game_x
        panda_Z = game_y
    """
    p = np.asarray(pos, dtype=float)
    return float(p[2]), float(p[0]), float(p[1])


def _nearest_time_indices(timestamps, query_times):
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    query_times = np.asarray(query_times, dtype=float).reshape(-1)
    if timestamps.size == 0:
        return np.zeros(query_times.size, dtype=int)

    right = np.searchsorted(timestamps, query_times, side="left")
    right = np.clip(right, 0, timestamps.size - 1)
    left = np.clip(right - 1, 0, timestamps.size - 1)
    use_left = np.abs(query_times - timestamps[left]) <= np.abs(timestamps[right] - query_times)
    return np.where(use_left, left, right).astype(int)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class EpisodeVideoRenderer(ShowBase):
    """
    Panda3D-based renderer that converts AimandShootEpisodeRecorder episodes
    into MP4 video files.

    Panda3D uses process-wide global state, so only one instance should exist
    per process and the resolution is fixed at construction time.

    Parameters
    ----------
    task : AimandShootSpiderShotTask
        Game environment instance used to replay trajectories.
    config : dict, optional
        Supported keys:
          width        (int)   output video width,  default 1920
          height       (int)   output video height, default 1080
          fps          (int)   frames per second,   default 60
          output_dir   (str)   directory for auto-named outputs, default "."
          target_color (tuple) RGBA colour for the target sphere, default white

    Usage
    -----
    renderer = EpisodeVideoRenderer(task, config={"fps": 60, "width": 1280, "height": 720})
    renderer.render_episode(ep_rec, "episode.mp4")
    renderer.render_episodes([ep_rec1, ep_rec2])          # auto-named outputs
    """

    def __init__(self, task: AnSTask, config: dict = None):
        global _PANDA_INITIALIZED

        self.cfg     = {**DEFAULT_RENDER_CONFIG, **(config or {})}
        self._width  = int(self.cfg["width"])
        self._height = int(self.cfg["height"])
        self._fps    = int(self.cfg["fps"])

        # Global Panda3D PRC config — must be applied before ShowBase.__init__().
        if not _PANDA_INITIALIZED:
            cache_dir = self.cfg.get("panda_cache_dir") or _default_panda_cache_dir()
            cache_dir = _ensure_panda_cache_dir(Path(cache_dir))
            loadPrcFileData("", "window-type offscreen")
            loadPrcFileData("", f"win-size {self._width} {self._height}")
            loadPrcFileData("", "sync-video 0")
            loadPrcFileData("", "load-display pandagl")
            loadPrcFileData("", f"model-cache-dir {_panda_prc_path(cache_dir)}")
            _PANDA_INITIALIZED = True

        super().__init__()
        self.disableMouse()
        self.camera.setPos(0, 0, 0)

        # Camera lens — FOV and near/far must match engine.py exactly.
        # Values come from the task config (same config both use).
        fov_w = float(task.camera_fov_deg[0])
        fov_h = float(task.camera_fov_deg[1])
        self.camLens.setFov(fov_w, fov_h)
        self.camLens.setNearFar(0.01, 2000.0)

        self.task_env = task

        self._setup_scene()
        self._setup_hud()

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self.setBackgroundColor(0.0, 0.0, 0.0, 1.0)

        # Lighting — matches engine.py
        ambient = AmbientLight("ambient")
        ambient.setColor((0.6, 0.6, 0.6, 1))
        self.render.setLight(self.render.attachNewNode(ambient))

        directional = DirectionalLight("directional")
        directional.setColor((0.6, 0.6, 0.6, 1))
        directional.setDirection(Vec3(-1, -1, -2))
        self.render.setLight(self.render.attachNewNode(directional))

        self._bg_sphere = draw_background_sphere(
            base=self,
            texture_path=TEXTURE_PATHS.get("default"),
            radius=500.0,
            lat_segments=32,
            lon_segments=64,
        )

        # Target sphere — radius=1.0; actual visual radius set via setScale per episode
        self._target_sphere = draw_solid_sphere(
            base=self,
            name="target",
            position=(0, 1, 0),   # placeholder; updated each frame
            radius=1.0,
            color=self.cfg["target_color"],
            unlit=True,
        )

        # Crosshair at screen centre (aspect2d)
        self._setup_crosshair()
        self._setup_gaze_overlay()

        # Render-to-texture for fast CPU frame capture.
        # Explicitly bind to the colour bitplane so we always get RGBA pixels.
        self._frame_tex = Texture("frame_capture")
        self.win.addRenderTexture(
            self._frame_tex,
            GraphicsOutput.RTM_copy_ram,
            GraphicsOutput.RTP_color,
        )

    def _setup_crosshair(self):
        crosshair_node = self.aspect2d.attachNewNode("crosshair")
        radius = float(self.cfg["crosshair_radius"])
        color = self.cfg["crosshair_color"]

        lines = LineSegs("crosshair_circle")
        lines.setColor(*color)
        lines.setThickness(float(self.cfg["crosshair_thickness"]))
        angles = np.linspace(0.0, 2.0 * np.pi, 65)
        lines.moveTo(radius, 0.0, 0.0)
        for theta in angles[1:]:
            lines.drawTo(float(radius * np.cos(theta)), 0.0, float(radius * np.sin(theta)))
        crosshair_node.attachNewNode(lines.create())

    def _setup_gaze_overlay(self):
        self._gaze_layer = self.aspect2d.attachNewNode("gaze_layer")
        self._gaze_marker = self._gaze_layer.attachNewNode("gaze_marker")
        self._gaze_marker.setTransparency(TransparencyAttrib.MAlpha)

        size = float(self.cfg["gaze_marker_size"])
        color = self.cfg["gaze_color"]

        marker = LineSegs("gaze_x")
        marker.setColor(*color)
        marker.setThickness(2.5)
        marker.moveTo(-size, 0.0, -size)
        marker.drawTo(size, 0.0, size)
        marker.moveTo(-size, 0.0, size)
        marker.drawTo(size, 0.0, -size)
        self._gaze_marker.attachNewNode(marker.create())

        self._gaze_marker.hide()
        self._gaze_trail_np = None

    # ------------------------------------------------------------------
    # HUD setup (top-left)
    # ------------------------------------------------------------------

    def _setup_hud(self):
        """Build top-left HUD skeleton; text is filled per episode."""
        aspect = self._width / self._height
        layer = self.aspect2d.attachNewNode("hud_layer")
        layer.setPos(-aspect, 0, 1.0)   # top-left anchor

        panel_w = min(1.1, 0.85 * 2.0 * aspect)
        panel_h = 1.15
        panel = CardMaker("hud_bg")
        panel.setFrame(0.0, panel_w, -panel_h, 0.0)
        panel_np = layer.attachNewNode(panel.generate())
        panel_np.setTransparency(TransparencyAttrib.MAlpha)
        panel_np.setColor(0.0, 0.0, 0.0, 0.55)

        title = TextNode("hud_title")
        title.setText("Player / Task State")
        title.setTextColor(1, 1, 1, 1)
        title_np = layer.attachNewNode(title)
        title_np.setScale(0.046)
        title_np.setPos(0.03, 0, -0.07)

        # Static body — player state + initial env + result
        self._hud_body_node = TextNode("hud_body")
        self._hud_body_node.setTextColor(0.80, 1.00, 0.80, 1)
        body_np = layer.attachNewNode(self._hud_body_node)
        body_np.setScale(0.034)
        body_np.setPos(0.03, 0, -0.16)

        # Dynamic: current time / TCT
        # Positioned below the static body (~16 lines × 0.044 units = 0.70 + 0.16 origin)
        self._hud_time_node = TextNode("hud_time")
        self._hud_time_node.setTextColor(1.00, 0.90, 0.50, 1)
        time_np = layer.attachNewNode(self._hud_time_node)
        time_np.setScale(0.040)
        time_np.setPos(0.03, 0, -0.96)

        # Dynamic: real-time target monitor position (light blue)
        self._hud_target_pos_node = TextNode("hud_target_pos")
        self._hud_target_pos_node.setTextColor(0.60, 0.85, 1.00, 1)
        tpos_np = layer.attachNewNode(self._hud_target_pos_node)
        tpos_np.setScale(0.034)
        tpos_np.setPos(0.03, 0, -1.03)

        # Static: manifold info (orange; visible only when manifold is active)
        self._hud_manifold_node = TextNode("hud_manifold")
        self._hud_manifold_node.setTextColor(1.00, 0.60, 0.10, 1)
        self._hud_manifold_node.setText("")   # empty until set_manifold_info() is called
        manifold_np = layer.attachNewNode(self._hud_manifold_node)
        manifold_np.setScale(0.034)
        manifold_np.setPos(0.03, 0, -1.10)

    def _hud_set_static(
        self,
        player_state: dict,
        init_game_env: dict,
        result_report: dict,
    ):
        """Populate static HUD body; called once at the start of each episode."""
        lines = []

        # Player state
        for k, v in player_state.items():
            if k == "head_position":
                hp = np.asarray(v, dtype=float).reshape(-1)
                lines.append(f"head_pos: ({hp[0]:.3f}, {hp[1]:.3f}, {hp[2]:.3f}) mm")
            elif isinstance(v, (float, np.floating)):
                lines.append(f"{k}: {v:.3f}")
            elif isinstance(v, np.ndarray) and v.size == 1:
                lines.append(f"{k}: {float(v.flat[0]):.3f}")
            else:
                lines.append(f"{k}: {v}")

        lines.append("")   # spacer

        # Initial camera
        cam = np.asarray(init_game_env.get("camera_azel_deg", [0.0, 0.0]))
        lines.append(f"cam init  az={cam[0]:.3f}°  el={cam[1]:.3f}°")

        # Target initial position
        tpos = np.asarray(init_game_env.get("target_pos_monitor_mm", [0.0, 0.0]))
        lines.append(f"target init  ({tpos[0]:.3f}, {tpos[1]:.3f}) mm")

        # Target geometry & motion
        lines.append(
            f"target  r={init_game_env.get('target_radius_deg', 0):.3f}°"
            f"  spd={init_game_env.get('target_speed_deg_s', 0):.3f} °/s"
        )
        orbit = np.asarray(init_game_env.get("target_orbit_axis_deg", [0.0, 0.0]))
        lines.append(f"orbit axis  az={orbit[0]:.3f}°  el={orbit[1]:.3f}°")
        lines.append(f"motion dir: {init_game_env.get('target_motion_dir_deg', 0.0):.3f}°")

        lines.append("")   # spacer

        # Episode result
        tct   = result_report.get(FEATURES.TCT,   0.0)
        err   = result_report.get(FEATURES.ERR,   0.0)
        res   = result_report.get(FEATURES.RES,   False)
        trunc = result_report.get(FEATURES.TRUNC, False)
        lines.append(
            f"result: {'HIT' if res else 'MISS'}  err={err:.3f} mm"
            + ("  [TRUNC]" if trunc else "")
        )

        self._hud_body_node.setText("\n".join(lines))

    def _hud_set_dynamic(self, t_ms: float, tct_ms: float, target_monitor_mm=None):
        self._hud_time_node.setText(f"t = {t_ms:.0f} ms  /  TCT = {tct_ms:.0f} ms")
        if target_monitor_mm is not None:
            mm = np.asarray(target_monitor_mm, dtype=float)
            self._hud_target_pos_node.setText(
                f"target monitor: ({mm[0]:+.3f}, {mm[1]:+.3f}) mm"
            )

    def set_manifold_info(
        self,
        manifold_name: Optional[str] = None,
        tau: Optional[float] = None,
        epoch: Optional[int] = None,
    ) -> None:
        """Display (or clear) manifold metadata in the HUD.

        Call before rendering a batch of episodes.  Pass all-None (or call
        with no arguments) to clear the manifold line.

        Parameters
        ----------
        manifold_name : str or None
        tau           : float or None
        epoch         : int or None — checkpoint epoch; None means "latest"
        """
        if manifold_name is None:
            self._hud_manifold_node.setText("")
        else:
            ep_str  = f"ep{epoch:04d}" if epoch is not None else "latest"
            tau_str = f"{tau:.3f}" if tau is not None else "?"
            self._hud_manifold_node.setText(
                f"manifold: {manifold_name}  ep={ep_str}  τ={tau_str}"
            )

    # ------------------------------------------------------------------
    # Camera & target helpers
    # ------------------------------------------------------------------

    def _apply_camera(self, camera_azel_deg):
        """
        Apply az/el to Panda3D camera.
        Convention is identical to engine.py: setHpr(-azim, elev, 0).
        """
        self.camera.setHpr(
            -float(camera_azel_deg[0]),
             float(camera_azel_deg[1]),
             0.0,
        )

    def _place_target(self, target_pos_world, visual_radius: float):
        """
        Place the target sphere at the correct Panda3D world position.

        Game world coordinates are normalized to the unit sphere, then
        remapped to Panda3D via (panda_X=game_z, panda_Y=game_x, panda_Z=game_y).
        The sphere scale encodes the angular radius.
        """
        p = np.asarray(target_pos_world, dtype=float)
        norm = np.linalg.norm(p)
        if norm > 1e-9:
            p /= norm
        px, py, pz = _game_to_panda(p)
        self._target_sphere.setScale(visual_radius)
        self._target_sphere.setPos(px, py, pz)

    def _monitor_mm_to_aspect2d(self, monitor_mm):
        mm = np.asarray(monitor_mm, dtype=float).reshape(2)
        half = np.asarray(self.task_env.monitor_half_size_mm, dtype=float)
        aspect = self._width / self._height
        return float(mm[0] / half[0] * aspect), float(mm[1] / half[1])

    def _update_gaze_overlay(self, gaze_mm=None, trail_mm=None):
        if self._gaze_trail_np is not None:
            self._gaze_trail_np.removeNode()
            self._gaze_trail_np = None

        if gaze_mm is None:
            self._gaze_marker.hide()
            return

        gx, gz = self._monitor_mm_to_aspect2d(gaze_mm)
        self._gaze_marker.setPos(gx, 0.0, gz)
        self._gaze_marker.show()

        if trail_mm is None:
            return
        trail = np.asarray(trail_mm, dtype=float).reshape(-1, 2)
        if trail.shape[0] < 2:
            return

        segs = LineSegs("gaze_trail")
        segs.setColor(*self.cfg["gaze_trail_color"])
        segs.setThickness(float(self.cfg["gaze_trail_thickness"]))
        x0, z0 = self._monitor_mm_to_aspect2d(trail[0])
        segs.moveTo(x0, 0.0, z0)
        for point in trail[1:]:
            x, z = self._monitor_mm_to_aspect2d(point)
            segs.drawTo(x, 0.0, z)
        self._gaze_trail_np = self._gaze_layer.attachNewNode(segs.create())
        self._gaze_trail_np.setTransparency(TransparencyAttrib.MAlpha)

    def _gaze_trail_for_time(self, gaze_history, t_ms):
        if not gaze_history:
            return None
        trail_ms = float(self.cfg["gaze_trail_ms"])
        return [
            point for ts, point in gaze_history
            if float(t_ms) - float(ts) <= trail_ms
        ]

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def _grab_frame_bgr(self) -> np.ndarray:
        """Render one frame and return a BGR uint8 array of shape (H, W, 3)."""
        self.graphicsEngine.renderFrame()
        data = self._frame_tex.getRamImageAs("RGB")
        arr = np.frombuffer(bytes(data), dtype=np.uint8).reshape(
            self._height, self._width, 3
        )
        # Panda3D stores rows bottom-to-top; flip rows and convert RGB → BGR
        return arr[::-1, :, ::-1].copy()

    # ------------------------------------------------------------------
    # Trajectory reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_trajectory(self, ep_rec: AimandShootEpisodeRecorder):
        """
        Upsample the recorded hand trajectory to 1ms, replay it through
        the task to obtain per-ms world state, then downsample to video FPS.

        Returns
        -------
        frames : list of (camera_azel_deg, target_pos_world, target_monitor_mm, gaze_mm, t_ms)
        tct_ms : float
        visual_radius : float   target sphere radius in Panda3D world units
        """
        init_cond = ep_rec.init_game_env
        tct_ms = float(ep_rec.result_report[FEATURES.TCT])

        # --- upsample keyframes → 1ms (OTG for MUSCLE-spaced segments) ---
        p_fine, v_fine, t_fine = ep_rec.get_trajectory(interpolate=True, episode_time=True)

        # --- reset task to initial conditions (t=0, before orbit) ---
        self.task_env.reset(
            camera_azel_deg=init_cond["camera_azel_deg"],
            target_pos_world=init_cond["target_pos_world"],
            target_orbit_axis_deg=init_cond["target_orbit_axis_deg"],
            target_speed_deg_s=init_cond["target_speed_deg_s"],
            target_radius_deg=init_cond["target_radius_deg"],
        )
        init_hand_pos_mm = self.task_env.hand_pos_mm.copy()
        p_abs = p_fine + init_hand_pos_mm   # episode-relative → absolute

        # --- replay → per-ms camera az/el, target world position, target monitor mm ---
        traj = self.task_env.replay_and_save(
            p_abs, v_fine,
            interval=TINTERVAL.INTERP1,
            unit='ms',
            keys=["camera_azel_deg", "target_pos_world", "target_pos_monitor_mm"],
        )
        cam_traj = traj["camera_azel_deg"]        # (N, 2)
        tgt_traj = traj["target_pos_world"]       # (N, 3)
        mon_traj = traj["target_pos_monitor_mm"]  # (N, 2)

        # --- downsample to video frame times ---
        ms_per_frame = 1000.0 / self._fps
        frame_times = np.arange(0.0, tct_ms + ms_per_frame, ms_per_frame)
        frame_times = frame_times[frame_times <= tct_ms + 1e-6]
        if frame_times.size == 0 or frame_times[-1] < tct_ms - 1e-6:
            frame_times = np.append(frame_times, tct_ms)
        indices = _nearest_time_indices(t_fine, frame_times)
        gaze_frame_traj = [None] * len(indices)
        if getattr(ep_rec, "has_gaze_metrics", lambda: False)():
            try:
                gaze_pos, _, gaze_ts = ep_rec.get_gaze_trajectory(interpolate=True)
                gaze_ts = np.asarray(gaze_ts, dtype=float).reshape(-1)
                gaze_pos = np.asarray(gaze_pos, dtype=float).reshape(-1, 2)
                if gaze_ts.size >= 1 and gaze_pos.shape[0] == gaze_ts.size:
                    frame_ts = frame_times.astype(float)
                    gaze_frame_traj = np.column_stack((
                        np.interp(frame_ts, gaze_ts, gaze_pos[:, 0]),
                        np.interp(frame_ts, gaze_ts, gaze_pos[:, 1]),
                    ))
            except (AssertionError, ValueError, KeyError):
                gaze_frame_traj = [None] * len(indices)

        frames = [
            (cam_traj[i], tgt_traj[i], mon_traj[i], None if gaze_frame_traj[j] is None else gaze_frame_traj[j], float(frame_times[j]))
            for j, i in enumerate(indices)
        ]

        # Always ensure the last frame is at the shoot moment (t_fine[-1] ≈ tct_ms).
        # Without this, 60fps downsampling can leave the last frame up to ~16ms early,
        # making the freeze-frame HUD show a different target position than what was
        # used to compute err, causing apparent contradictions like |x| > err.
        last_idx = len(t_fine) - 1
        if indices[-1] < last_idx:
            last_gaze = None
            if getattr(ep_rec, "has_gaze_metrics", lambda: False)():
                try:
                    gaze_pos, _, gaze_ts = ep_rec.get_gaze_trajectory(interpolate=True)
                    gaze_ts = np.asarray(gaze_ts, dtype=float).reshape(-1)
                    gaze_pos = np.asarray(gaze_pos, dtype=float).reshape(-1, 2)
                    if gaze_ts.size >= 1 and gaze_pos.shape[0] == gaze_ts.size:
                        last_t = float(t_fine[last_idx])
                        last_gaze = np.array([
                            np.interp(last_t, gaze_ts, gaze_pos[:, 0]),
                            np.interp(last_t, gaze_ts, gaze_pos[:, 1]),
                        ], dtype=float)
                except (AssertionError, ValueError, KeyError):
                    last_gaze = None
            frames.append((cam_traj[last_idx], tgt_traj[last_idx], mon_traj[last_idx], last_gaze, float(t_fine[last_idx])))

        # --- target visual radius: chord on unit sphere ≈ radians for small angles ---
        visual_radius = float(np.radians(init_cond.get("target_radius_deg", 0.05)))

        return frames, tct_ms, visual_radius

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_episode(
        self,
        ep_rec: AimandShootEpisodeRecorder,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Render a single episode to an MP4 video file.

        Parameters
        ----------
        ep_rec : AimandShootEpisodeRecorder
            A completed (done) episode record.
        output_path : str, optional
            Full path for the output file.  If None, an auto-name is
            generated inside config["output_dir"].

        Returns
        -------
        str : path of the written video file.
        """
        if output_path is None:
            out_dir = Path(self.cfg["output_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            tct = ep_rec.result_report.get(FEATURES.TCT, 0)
            tag = "hit" if ep_rec.result_report.get(FEATURES.RES, False) else "miss"
            from ..utils.myutils import get_compact_timestamp_str
            output_path = str(out_dir / f"{get_compact_timestamp_str()}_tct{tct:.0f}_{tag}.mp4")

        frames_data, tct_ms, visual_radius = self._reconstruct_trajectory(ep_rec)

        self._hud_set_static(
            ep_rec.player_state,
            ep_rec.init_game_env,
            ep_rec.result_report,
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, self._fps, (self._width, self._height))
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter could not open: {output_path}")

        gaze_history = []
        for cam_azel, tgt_world, tgt_monitor_mm, gaze_mm, t_ms in frames_data:
            self._apply_camera(cam_azel)
            self._place_target(tgt_world, visual_radius)
            self._hud_set_dynamic(t_ms, tct_ms, tgt_monitor_mm)
            if gaze_mm is not None:
                gaze_history.append((t_ms, np.asarray(gaze_mm, dtype=float)))
            self._update_gaze_overlay(gaze_mm, self._gaze_trail_for_time(gaze_history, t_ms))
            writer.write(self._grab_frame_bgr())

        # --- hold final frame for freeze_ms after click ---
        freeze_n = int(round(self._fps * self.cfg["freeze_ms"] / 1000.0))
        if freeze_n > 0 and frames_data:
            last_cam, last_tgt, last_mon, last_gaze, _ = frames_data[-1]
            self._apply_camera(last_cam)
            self._place_target(last_tgt, visual_radius)
            self._hud_set_dynamic(tct_ms, tct_ms, last_mon)
            self._update_gaze_overlay(last_gaze, self._gaze_trail_for_time(gaze_history, tct_ms))
            still = self._grab_frame_bgr()
            for _ in range(freeze_n):
                writer.write(still)

        writer.release()
        print(f"[Renderer] Saved {len(frames_data)} + {freeze_n} freeze frames → {output_path}")
        return output_path

    def render_all(
        self,
        ep_recs: List[AimandShootEpisodeRecorder],
        output_path: str,
    ) -> str:
        """
        Render all episodes into one combined MP4 video file.

        Episodes are concatenated in order; each episode ends with a
        freeze-frame hold (config key ``freeze_ms``).

        Parameters
        ----------
        ep_recs : list of AimandShootEpisodeRecorder
        output_path : str
            Full path for the output file.

        Returns
        -------
        str : path of the written video file.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, self._fps, (self._width, self._height))
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter could not open: {output_path}")

        freeze_n = int(round(self._fps * self.cfg["freeze_ms"] / 1000.0))
        total_frames = 0

        for ep_rec in ep_recs:
            frames_data, tct_ms, visual_radius = self._reconstruct_trajectory(ep_rec)
            self._hud_set_static(ep_rec.player_state, ep_rec.init_game_env, ep_rec.result_report)

            gaze_history = []
            for cam_azel, tgt_world, tgt_monitor_mm, gaze_mm, t_ms in frames_data:
                self._apply_camera(cam_azel)
                self._place_target(tgt_world, visual_radius)
                self._hud_set_dynamic(t_ms, tct_ms, tgt_monitor_mm)
                if gaze_mm is not None:
                    gaze_history.append((t_ms, np.asarray(gaze_mm, dtype=float)))
                self._update_gaze_overlay(gaze_mm, self._gaze_trail_for_time(gaze_history, t_ms))
                writer.write(self._grab_frame_bgr())
            total_frames += len(frames_data)

            if freeze_n > 0 and frames_data:
                last_cam, last_tgt, last_mon, last_gaze, _ = frames_data[-1]
                self._apply_camera(last_cam)
                self._place_target(last_tgt, visual_radius)
                self._hud_set_dynamic(tct_ms, tct_ms, last_mon)
                self._update_gaze_overlay(last_gaze, self._gaze_trail_for_time(gaze_history, tct_ms))
                still = self._grab_frame_bgr()
                for _ in range(freeze_n):
                    writer.write(still)
                total_frames += freeze_n

        writer.release()
        print(f"[Renderer] {len(ep_recs)} episodes, {total_frames} frames → {output_path}")
        return output_path

    def render_episodes(
        self,
        ep_recs: List[AimandShootEpisodeRecorder],
        output_paths: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Render a list of episodes sequentially.

        Parameters
        ----------
        ep_recs : list of AimandShootEpisodeRecorder
        output_paths : list of str, optional
            One output path per episode.  Pass None for any entry (or omit
            the list entirely) to use auto-generated names under
            config["output_dir"].

        Returns
        -------
        list of str : paths of written video files (same order as ep_recs).
        """
        if output_paths is not None and len(output_paths) != len(ep_recs):
            raise ValueError("output_paths length must match ep_recs length.")

        if output_paths is None:
            output_paths = [None] * len(ep_recs)

        out_dir = Path(self.cfg["output_dir"])
        results = []
        for idx, (ep_rec, path) in enumerate(zip(ep_recs, output_paths)):
            if path is None:
                out_dir.mkdir(parents=True, exist_ok=True)
                tct = ep_rec.result_report.get(FEATURES.TCT, 0)
                tag = "hit" if ep_rec.result_report.get(FEATURES.RES, False) else "miss"
                from ..utils.myutils import get_compact_timestamp_str
                path = str(out_dir / f"{idx:04d}_{get_compact_timestamp_str()}_tct{tct:.0f}_{tag}.mp4")
            results.append(self.render_episode(ep_rec, output_path=path))

        return results
