# Terminal 3D Renderer

Colourful real-time 3D rendering directly inside your terminal. The project projects and shades triangle meshes, writes ANSI 256-colour frames, and is tuned to run smoothly on plain Debian GNU/Linux shells (tested on Debian 12, kernel 6.1 on ARM64).

## Highlights

- Pure Python 3.10+ implementation with no third-party dependencies.
- Software rasteriser with back-face culling, depth buffering, and diffuse shading.
- Vibrant ANSI 256-colour output with adaptive character shading.
- High-resolution glyph gradient (ASCII + Unicode) for finer detail in the terminal.
- Respects terminal resizes on the fly; just stretch the window for more pixels.
- Easily customisable rotation speed, FOV, light direction, and render duration.
- Checkerboard ground plane with dynamic shadows, floor reflections, and enhanced lighting for depth cues.
- Integrated HUD overlay (FPS counter in the top-right corner by default).
- Includes a Cornell Box scene for lighting/material evaluation.

## Requirements

- Python 3.10 or newer.
- A terminal that supports ANSI escape sequences (most modern shells, including Debian's default `bash`/`dash`/`zsh`).

## Setup

After cloning, you can bootstrap everything with `make` (uses Python 3.10+):

```bash
make install
```

This creates a `.venv/` virtual environment and installs the package in editable mode. If you prefer to manage environments yourself, run `pip install -e .` inside whatever Python environment you choose.

## Run the demo

With the virtual environment prepared, launch the animation:

```bash
make run
```

The `run` target simply calls the installed console script `terminal-renderer`. To tweak parameters manually you can invoke it directly:

```bash
.venv/bin/terminal-renderer --fps 24 --fov 70 --scale 2.0
```

Available flags:

| Flag | Description | Default |
| --- | --- | --- |
| `--fps` | Target frames per second | `20` |
| `--fov` | Field of view in degrees | `70` |
| `--distance` | Camera distance from mesh centre | `5` |
| `--light X Y Z` | Directional light vector | `-0.4 0.8 -0.6` |
| `--object` | Scene to render (`cube`, `cornell`) | `cube` |
| `--scale` | Uniform mesh scale multiplier | `2.0` |
| `--speed` | Rotation speed multiplier | `1.0` |
| `--frames` | Render a fixed number of frames before exiting | `0` (run forever) |
| `--no-floor` | Disable the checkerboard floor | `false` |
| `--no-shadows` | Disable projected cube shadows | `false` |
| `--no-reflections` | Disable reflective floor rendering | `false` |
| `--floor-size` | Edge length of the floor plane (world units) | `12.0` |
| `--floor-tiles` | Checkerboard tiles per side (higher = finer detail) | `10` |
| `--gpu` | Experimental DRM/KMS output (requires root + libdrm, falls back if unavailable) | disabled |
| `--pixel-step` | Render every Nth pixel and fill the gaps (higher = faster, lower detail) | `1` |
| `--reflection-depth` | Maximum recursive reflection depth | `2` |
| `--ambient` | Ambient light strength | `0.22` |
| `--gi-strength` | Sky/ground global illumination multiplier | `0.35` |
| `--sun-intensity` | Sun disc intensity | `1.4` |

Feel free to resize the terminal while the renderer is running; the engine automatically adapts to the new resolution.

### Experimental GPU output (DRM/KMS)

If your machine exposes a `/dev/dri/card*` node and you have `libdrm` installed, you can request the experimental direct-to-GPU presenter:

```bash
.venv/bin/terminal-renderer --object cornell --gpu
```

This path tries to set the process as the DRM master, creates a dumb buffer, and scans it out through the primary CRTC. The renderer renders into a matrix instead of ANSI strings, which removes most of the terminal overhead. The feature currently requires root (or appropriate udev rules) and only targets simple XRGB8888 framebuffers; if initialisation fails the program falls back to the regular terminal output and prints a warning.

When GPU mode is enabled without explicit quality flags, the renderer automatically bumps `--pixel-step` to `2` and caps `--reflection-depth` to `1` to keep frame times manageable on low-power hardware. You can still override either value manually.

The sky gradient, sun bloom, and fake global illumination can be tuned via `--sun-intensity`, `--ambient`, and `--gi-strength`. Lower these values for a flatter look, or raise them for brighter outdoor lighting.

## Tests

All logic-level tests use Python's built-in `unittest` runner.

```bash
make test
```

## Debian 12 (ARM64) quickstart

The renderer was tuned on Debian GNU/Linux 12 (bookworm) running on AArch64, e.g. `Linux localhost 6.1.0-34-avf-arm64 #1 SMP Debian 6.1.135-1 (2025-04-25) aarch64`. Follow these steps after logging in as your regular user:

1. Install the base tooling (Git, Python, venv, Make):

	```bash
	sudo apt update
	sudo apt install -y git python3 python3-venv python3-pip make
	```

2. Clone the repository and change into it:

	```bash
	git clone https://github.com/<your-account>/terminal-3d-renderer.git
	cd terminal-3d-renderer
	```

3. Create the virtual environment and install the package in editable mode:

	```bash
	make install
	```

4. Launch the rotating cube demo (Ctrl+C to stop):

	```bash
	make run
	```

5. Optional maintenance commands:

	```bash
	make test      # run the unit tests
	make package   # build wheel + sdist into dist/
	make distclean # remove .venv and temporary artifacts
	```

Feel free to resize the terminal while it runs; the engine adapts automatically.

## Build distribution artifacts

Generate wheel and sdist packages via:

```bash
make package
```

## How it works

- `src/renderer/engine.py` implements vector math, perspective projection, triangle rasterisation, a simple diffuse lighting model, and ANSI colour mapping.
- `src/renderer/objects.py` contains ready-made meshes (colourful cube, Cornell box) described via triangles.
- `src/renderer/terminal.py` hides cursor flicker, handles clearing, and streams frames to the terminal.
- `src/main.py` glues everything together into a CLI-driven animation loop with adaptive frame timing.

The renderer uses a z-buffer to resolve visibility, back-face culling to skip hidden surfaces, and a small ASCII gradient to add depth perception in addition to colours.

Enjoy exploring and tweaking the parameters to craft your own terminal art!
