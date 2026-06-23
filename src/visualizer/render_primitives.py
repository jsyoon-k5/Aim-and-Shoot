import math
import re
from pathlib import Path

import cv2
import numpy as np
from panda3d.core import (
	CompassEffect,
	CullFaceAttrib,
	Geom,
	GeomNode,
	GeomTriangles,
	GeomVertexData,
	GeomVertexFormat,
	GeomVertexWriter,
	Texture,
)


def _panda_os_path(path) -> str:
	"""Return a path string Panda3D can load on native Windows Conda."""
	text = str(path)
	match = re.match(r"^/([A-Za-z])/(.*)$", text)
	if match:
		return f"{match.group(1)}:/{match.group(2)}"
	return text


TEXTURE_PATHS = {
	"default": _panda_os_path(Path(__file__).parent.parent.parent.resolve() / "assets" / "textures" / "defaultblack.png"),
}


def _load_texture_from_image(path) -> Texture | None:
	img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
	if img is None:
		return None

	if img.ndim == 2:
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		panda_format = Texture.FRgb
	elif img.shape[2] == 3:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		panda_format = Texture.FRgb
	elif img.shape[2] == 4:
		img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
		panda_format = Texture.FRgba
	else:
		return None

	# Panda3D reads raw texture memory bottom-to-top here, while cv2 returns
	# image rows top-to-bottom.
	img = np.ascontiguousarray(np.flipud(img))
	height, width = img.shape[:2]
	texture = Texture(Path(path).name)
	texture.setup2dTexture(width, height, Texture.TUnsignedByte, panda_format)
	texture.setRamImage(img.tobytes())
	return texture


def create_uv_sphere_node(name: str, radius: float, lat_segments: int, lon_segments: int):
	"""Create and return a UV sphere GeomNode wrapped in a NodePath."""
	fmt = GeomVertexFormat.getV3t2()
	vdata = GeomVertexData(name, fmt, Geom.UHStatic)
	vdata.setNumRows((lat_segments + 1) * (lon_segments + 1))

	v_writer = GeomVertexWriter(vdata, "vertex")
	t_writer = GeomVertexWriter(vdata, "texcoord")

	for lat in range(lat_segments + 1):
		v = lat / lat_segments
		phi = math.pi * v
		sin_phi = math.sin(phi)
		cos_phi = math.cos(phi)

		for lon in range(lon_segments + 1):
			u = lon / lon_segments
			theta = 2.0 * math.pi * u
			sin_theta = math.sin(theta)
			cos_theta = math.cos(theta)

			x = radius * sin_phi * sin_theta
			y = radius * sin_phi * cos_theta
			z = radius * cos_phi
			v_writer.addData3(x, y, z)

			# Flip V so the top of source image maps to upper hemisphere.
			t_writer.addData2(u, 1.0 - v)

	tris = GeomTriangles(Geom.UHStatic)
	for lat in range(lat_segments):
		for lon in range(lon_segments):
			i0 = lat * (lon_segments + 1) + lon
			i1 = i0 + 1
			i2 = i0 + (lon_segments + 1)
			i3 = i2 + 1

			tris.addVertices(i0, i2, i1)
			tris.addVertices(i1, i2, i3)

	geom = Geom(vdata)
	geom.addPrimitive(tris)
	node = GeomNode(name)
	node.addGeom(geom)
	return node


def draw_background_sphere(base, texture_path: str, radius: float = 500.0, lat_segments: int = 32, lon_segments: int = 64):
	"""Attach a textured background sphere to render and return its NodePath.

	Returns None if the texture cannot be loaded.
	"""
	if not texture_path:
		return None

	if not Path(texture_path).exists():
		print(f"[Background] Missing texture: {texture_path}")
		return None

	texture = _load_texture_from_image(texture_path)
	if not texture:
		print(f"[Background] Failed to load texture: {texture_path}")
		return None

	node = create_uv_sphere_node(
		name="background_sphere",
		radius=radius,
		lat_segments=lat_segments,
		lon_segments=lon_segments,
	)
	sphere = base.render.attachNewNode(node)
	sphere.setEffect(CompassEffect.make(base.camera, CompassEffect.P_pos))
	sphere.setBin("background", 0)
	sphere.setDepthWrite(False)
	sphere.setDepthTest(False)
	sphere.setLightOff()
	sphere.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))
	sphere.setTwoSided(True)
	sphere.setColor(1, 1, 1, 1)
	sphere.setTexture(texture, 1)
	return sphere


def draw_solid_sphere(
	base,
	name: str,
	position,
	radius: float = 0.15,
	color=(1.0, 0.2, 0.2, 1.0),
	lat_segments: int = 20,
	lon_segments: int = 40,
	unlit: bool = True,
):
	"""Attach a solid-color sphere to render and return its NodePath."""
	node = create_uv_sphere_node(
		name=name,
		radius=radius,
		lat_segments=lat_segments,
		lon_segments=lon_segments,
	)
	np = base.render.attachNewNode(node)
	pos = tuple(float(v) for v in position)
	np.setPos(*pos)
	np.setColor(*color)
	np.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))
	np.setTwoSided(True)
	if unlit:
		np.setLightOff()
	return np
