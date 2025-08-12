# ui/vispy_app.py
"""
VisPy interactive GPU-based black hole viewer.
Step 2.1: Minimal shader with disk + shadow (no lensing yet).
Run:
    python -m ui.vispy_app
"""

import numpy as np
from vispy import app, gloo

# GLSL fragment shader (placeholder disk + shadow)
FRAG_SHADER = """
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_fov;         // field of view in degrees
uniform vec3 u_cam_pos;      // camera position in world space
uniform vec3 u_cam_dir;      // camera direction (unit vector)
uniform vec3 u_cam_right;    // right vector
uniform vec3 u_cam_up;       // up vector

void main() {
    // Normalized pixel coords (0 to 1)
    vec2 uv = (gl_FragCoord.xy / u_resolution) * 2.0 - 1.0;
    uv.x *= u_resolution.x / u_resolution.y; // aspect correction

    // Simple ray direction from camera basis
    float fov_rad = radians(u_fov);
    vec3 ray_dir = normalize(u_cam_dir +
                             uv.x * tan(fov_rad / 2.0) * u_cam_right +
                             uv.y * tan(fov_rad / 2.0) * u_cam_up);

    // Fake black hole shadow: if ray_dir is near +Z axis, draw black
    float shadow = smoothstep(0.05, 0.02, length(ray_dir.xy));

    // Fake accretion disk in XY plane at Z=0
    float disk_mask = step(0.0, -ray_dir.y); // crude placeholder
    vec3 disk_color = vec3(1.0, 0.7, 0.3) * disk_mask;

    vec3 color = mix(disk_color, vec3(0.0), shadow);
    gl_FragColor = vec4(color, 1.0);
}
"""

# Pass-through vertex shader
VERT_SHADER = """
attribute vec2 a_position;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""


class BlackHoleCanvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(
            self, title="Black Hole Sim (GPU)", size=(800, 800), keys="interactive"
        )

        # Fullscreen quad vertices
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program["a_position"] = gloo.VertexBuffer(
            np.array([[-1, -1], [-1, +1], [+1, -1], [+1, +1]], dtype=np.float32)
        )
        self.indices = gloo.IndexBuffer(np.array([0, 1, 2, 1, 2, 3], dtype=np.uint32))

        # Camera parameters
        self.cam_pos = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        self.cam_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.cam_right = np.cross(self.cam_dir, self.cam_up)
        self.fov = 45.0
        self.time = 0.0

        gloo.set_state(
            clear_color="black",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )

        self.timer = app.Timer("auto", connect=self.on_timer, start=True)

        self.show()

    def on_draw(self, event):
        gloo.clear()
        self.program["u_resolution"] = self.size
        self.program["u_time"] = self.time
        self.program["u_fov"] = self.fov
        self.program["u_cam_pos"] = self.cam_pos
        self.program["u_cam_dir"] = self.cam_dir
        self.program["u_cam_up"] = self.cam_up
        self.program["u_cam_right"] = self.cam_right
        self.program.draw("triangles", self.indices)

    def on_timer(self, event):
        self.time += event.dt
        self.update()

    def on_mouse_wheel(self, event):
        self.fov = np.clip(self.fov - event.delta[1], 20.0, 90.0)

    def on_mouse_move(self, event):
        if event.is_dragging and event.buttons[0] == 1:
            dx, dy = event.delta
            sensitivity = 0.005
            yaw = dx * sensitivity
            pitch = -dy * sensitivity

            # Simple yaw-pitch camera rotation
            rot_yaw = self._rotation_matrix(self.cam_up, yaw)
            rot_pitch = self._rotation_matrix(self.cam_right, pitch)
            self.cam_dir = rot_pitch @ (rot_yaw @ self.cam_dir)
            self.cam_right = np.cross(self.cam_dir, self.cam_up)

    @staticmethod
    def _rotation_matrix(axis, theta):
        axis = axis / np.linalg.norm(axis)
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        return np.array(
            [
                [
                    a * a + b * b - c * c - d * d,
                    2 * (b * c - a * d),
                    2 * (b * d + a * c),
                ],
                [
                    2 * (b * c + a * d),
                    a * a + c * c - b * b - d * d,
                    2 * (c * d - a * b),
                ],
                [
                    2 * (b * d - a * c),
                    2 * (c * d + a * b),
                    a * a + d * d - b * b - c * c,
                ],
            ],
            dtype=np.float32,
        )


if __name__ == "__main__":
    c = BlackHoleCanvas()
    app.run()
