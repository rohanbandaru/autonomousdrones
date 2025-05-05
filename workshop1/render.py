"""
Render a 3D object with a pose from a CSV file using OpenGL and Pyglet.
----------------------------------
Dependencies:  pip install pyglet pyrr numpy
CSV format  :  time_ms, w, x, y, z – first row is a header
"""
import bisect, csv, ctypes, math, time
from pathlib import Path
import numpy as np
import pyglet
from pyglet import gl
from pyrr import quaternion, Matrix44

CSV_FILE = Path("workshop1/data/ori_1.csv")
CAMERA_Z = 3.6                        # 10 % closer than 4.0
FOV      = 55.0                       # degrees

# ────────────────────────── CSV                                     ─────────
def load_track(path: Path):
    """Read [ms, w, x, y, z] rows → (times_s, quats_xyzw)."""
    times, quats = [], []
    with path.open(newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            times.append(float(row["time_ms"]) * 0.001)   # to seconds
            # CSV stores w,x,y,z – pyrr expects x,y,z,w
            w, x, y, z = (float(row[k]) for k in ("w", "x", "y", "z"))
            quats.append(np.array([x, y, z, w], dtype="f4"))
    dur = times[-1] - times[0]
    # re‑base so first time stamp is 0 s
    times = [t - times[0] for t in times]
    return np.asarray(times, dtype="f4"), np.stack(quats), dur

KEY_T, KEY_Q, CLIP_LEN = load_track(CSV_FILE)   # pre‑loaded & immutable

# ────────────────────────── GL helpers                              ─────────
def _compile(src: bytes, stype):
    sh = gl.glCreateShader(stype)
    buf = ctypes.create_string_buffer(src)
    ptr = ctypes.cast(ctypes.pointer(ctypes.pointer(buf)),
                      ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
    ln  = ctypes.c_int(len(src))
    gl.glShaderSource(sh, 1, ptr, ctypes.byref(ln))
    gl.glCompileShader(sh)
    ok  = gl.GLint()
    gl.glGetShaderiv(sh, gl.GL_COMPILE_STATUS, ctypes.byref(ok))
    if not ok:
        log_len = gl.GLint()
        gl.glGetShaderiv(sh, gl.GL_INFO_LOG_LENGTH, ctypes.byref(log_len))
        log = ctypes.create_string_buffer(log_len.value)
        gl.glGetShaderInfoLog(sh, log_len, None, log)
        raise RuntimeError(log.value.decode())
    return sh

def _link(vs, fs):
    prog = gl.glCreateProgram()
    for s in (vs, fs): gl.glAttachShader(prog, s)
    gl.glLinkProgram(prog)
    ok = gl.GLint()
    gl.glGetProgramiv(prog, gl.GL_LINK_STATUS, ctypes.byref(ok))
    if not ok:
        log_len = gl.GLint()
        gl.glGetProgramiv(prog, gl.GL_INFO_LOG_LENGTH, ctypes.byref(log_len))
        log = ctypes.create_string_buffer(log_len.value)
        gl.glGetProgramInfoLog(prog, log_len, None, log)
        raise RuntimeError(log.value.decode())
    for s in (vs, fs): gl.glDeleteShader(s)
    return prog

VS = b"""
#version 330 core
layout(location=0) in vec3 pos;
layout(location=1) in vec3 col;
uniform mat4 u_mvp;
out vec3 v_col;
void main(){ v_col = col; gl_Position = u_mvp * vec4(pos,1.); }
"""
FS = b"""
#version 330 core
in vec3 v_col; out vec4 frag;
void main(){ frag = vec4(v_col,1.); }
"""

# ────────────────────────── geometry (same as before)               ─────────
hx, hy, hz = 0.36, 0.735, 0.039
verts = np.array([
    -hx, -hy, -hz,  .5,.5,.5,   hx, -hy, -hz,  .6,.6,.6,
     hx,  hy, -hz,  .7,.7,.7,  -hx,  hy, -hz,  .8,.8,.8,
    -hx, -hy,  hz,  1,0,0,      hx, -hy,  hz,  0,1,0,
     hx,  hy,  hz,  0,0,1,     -hx,  hy,  hz,  1,1,0,
], dtype="f4")
idx = np.array([
    0,1,2, 2,3,0,   4,5,6, 6,7,4,   0,4,7, 7,3,0,
    1,5,6, 6,2,1,   3,2,6, 6,7,3,   0,1,5, 5,4,0
], dtype="u4")

# ────────────────────────── GL setup                                 ────────
cfg = gl.Config(double_buffer=True, major_version=3, minor_version=3)
win = pyglet.window.Window(900, 600, "CSV‑driven device pose", config=cfg)
win.set_minimum_size(400, 300)

prog  = _link(_compile(VS, gl.GL_VERTEX_SHADER), _compile(FS, gl.GL_FRAGMENT_SHADER))
u_mvp = gl.glGetUniformLocation(prog, b"u_mvp")

vao, vbo, ebo = gl.GLuint(), gl.GLuint(), gl.GLuint()
gl.glGenVertexArrays(1, vao)
gl.glGenBuffers(1, vbo)
gl.glGenBuffers(1, ebo)
gl.glBindVertexArray(vao)

gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes,
                verts.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                gl.GL_STATIC_DRAW)
gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes,
                idx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
                gl.GL_STATIC_DRAW)

stride = 6 * 4
gl.glEnableVertexAttribArray(0)
gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE,
                         stride, ctypes.c_void_p(0))
gl.glEnableVertexAttribArray(1)
gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE,
                         stride, ctypes.c_void_p(12))

gl.glEnable(gl.GL_DEPTH_TEST)

# ────────────────────────── animation: CSV playback                 ─────────
start_real = time.perf_counter()

def current_quat():
    """Return the SLERP‑interpolated quaternion for *now* (loops)."""
    t = (time.perf_counter() - start_real) % CLIP_LEN
    i = bisect.bisect_right(KEY_T, t) - 1   # prev index
    j = (i + 1) % len(KEY_T)
    t0, t1 = KEY_T[i], KEY_T[j] if j else CLIP_LEN
    q0, q1 = KEY_Q[i], KEY_Q[j]
    if t1 == t0: return q0
    alpha = (t - t0) / (t1 - t0)
    # pyrr.quaternion.slerp performs spherical interpolation
    return quaternion.slerp(q0, q1, alpha)   # pyrr docs :contentReference[oaicite:0]{index=0}

# ────────────────────────── window events & draw loop               ─────────
@win.event
def on_key_press(sym, mods):
    if sym == pyglet.window.key.ESCAPE:
        win.close()
    if sym == pyglet.window.key.SPACE:
        global start_real
        start_real = time.perf_counter()   # restart playback

@win.event
def on_draw():
    win.clear()
    gl.glUseProgram(prog)

    proj = Matrix44.perspective_projection(FOV, win.width / win.height, 0.05, 100)
    view = Matrix44.look_at([0, 0, CAMERA_Z], [0, 0, 0], [0, 1, 0])
    model = Matrix44.from_quaternion(current_quat())
    mvp   = proj * view * model
    gl.glUniformMatrix4fv(u_mvp, 1, gl.GL_FALSE,
                          (gl.GLfloat * 16)(*mvp.astype("f4").flatten()))
    gl.glDrawElements(gl.GL_TRIANGLES, len(idx),
                      gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))

pyglet.app.run()
