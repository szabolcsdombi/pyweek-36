import pickle
import struct

import glm
import numpy as np
import pyglet
import zengl

pyglet.options['shadow_window'] = False
pyglet.options['debug_gl'] = False


KEYS = {
    1: 'Mouse1',
    2: 'Mouse3',
    4: 'Mouse2',
}

for c in '0123456789abcdefghijklmnopqrstuvwxyz':
    KEYS[ord(c)] = 'Key' + c.upper()


class PygletWindow(pyglet.window.Window):
    def __init__(self, width, height):
        self.alive = True
        self.mouse = (0, 0)
        self.prev_keys = set()
        self.keys = set()
        config = pyglet.gl.Config(
            major_version=3,
            minor_version=3,
            forward_compatible=True,
            double_buffer=True,
            depth_size=0,
            samples=0,
        )
        super().__init__(width=width, height=height, config=config, vsync=True)

    def on_resize(self, width, height):
        pass

    def on_draw(self):
        pass

    def on_key_press(self, symbol, modifiers):
        if key := KEYS.get(symbol):
            self.keys.add(key)

    def on_key_release(self, symbol, modifiers):
        if key := KEYS.get(symbol):
            self.keys.discard(key)

    def on_mouse_press(self, x, y, button, modifiers):
        self.mouse = x, y
        if key := KEYS.get(button):
            self.keys.add(key)

    def on_mouse_release(self, x, y, button, modifiers):
        self.mouse = x, y
        if key := KEYS.get(button):
            self.keys.discard(key)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.mouse = x, y

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse = x, y

    def on_close(self):
        self.alive = False

    def update(self):
        self.prev_keys = self.keys
        self.keys = self.keys.copy()
        self.flip()
        self.dispatch_events()
        return self.alive


class Window:
    def __init__(self):
        self._wnd = PygletWindow(1280, 720)
        self.size = self._wnd.get_framebuffer_size()
        self.aspect = 16.0 / 9.0
        self.loader = None

    @property
    def mouse(self):
        return self._wnd.mouse

    def key_down(self, key):
        return key in self._wnd.keys

    def update(self):
        pass

    def alive(self):
        return self._wnd.update()


window = Window()

ctx = zengl.context(window.loader)

image = ctx.image(window.size, 'rgba8unorm', texture=False)
depth = ctx.image(window.size, 'depth24plus', texture=False)
image.clear_value = (0.0, 0.0, 0.0, 1.0)

with open('assets/assets.pickle', 'rb') as f:
    assets = pickle.load(f)

vertex_buffer = ctx.buffer(assets['VertexData'])

uniform_buffer = ctx.buffer(size=64)


def make_pipeline(vertex_index, vertex_count):
    return ctx.pipeline(
        vertex_shader='''
            #version 330 core

            vec3 qtransform(vec4 q, vec3 v) {
                return v + 2.0 * cross(cross(v, q.xyz) - q.w * v, q.xyz);
            }

            layout (std140) uniform Common {
                mat4 mvp;
            };

            uniform vec3 position;
            uniform vec4 rotation;

            layout (location = 0) in vec3 in_vertex;
            layout (location = 1) in vec3 in_normal;
            layout (location = 2) in vec3 in_color;

            out vec3 v_vertex;
            out vec3 v_normal;
            out vec3 v_color;

            void main() {
                v_vertex = position + qtransform(rotation, in_vertex);
                v_normal = qtransform(rotation, in_normal);
                v_color = in_color;
                gl_Position = mvp * vec4(v_vertex, 1.0);
            }
        ''',
        fragment_shader='''
            #version 330 core

            in vec3 v_vertex;
            in vec3 v_normal;
            in vec3 v_color;

            layout (location = 0) out vec4 out_color;

            void main() {
                vec3 light = vec3(4.0, 3.0, 10.0);
                float lum = dot(normalize(light), normalize(v_normal)) * 0.7 + 0.3;
                out_color = vec4(pow(v_color * lum, vec3(1.0 / 2.2)), 1.0);
            }
        ''',
        layout=[
            {
                'name': 'Common',
                'binding': 0,
            },
        ],
        resources=[
            {
                'type': 'uniform_buffer',
                'binding': 0,
                'buffer': uniform_buffer,
            },
        ],
        uniforms={
            'position': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0, 1.0],
        },
        framebuffer=[image, depth],
        topology='triangles',
        cull_face='back',
        vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 3f', 0, 1, 2),
        first_vertex=vertex_index,
        vertex_count=vertex_count,
    )


pipelines = {
    name: make_pipeline(vertex_offset, vertex_count)
    for name, vertex_offset, vertex_count in assets['Objects']
}


def rotate(forward, upward, yaw, pitch, roll):
    upward = glm.angleAxis(roll, forward) * upward
    forward = glm.angleAxis(-yaw, upward) * forward
    sideways = glm.normalize(glm.cross(forward, upward))
    forward = glm.angleAxis(pitch, sideways) * forward
    upward = glm.normalize(glm.cross(sideways, forward))
    return forward, upward


def quat_look_at(forward, upward):
    forward = glm.normalize(forward)
    sideways = glm.normalize(glm.cross(forward, upward))
    upward = glm.normalize(glm.cross(sideways, forward))
    basis = glm.mat3(-sideways, -forward, upward)
    return glm.quat_cast(basis)


class SpaceShip:
    def __init__(self):
        self.position = glm.vec3(0.0, 0.0, 0.0)
        self.forward = glm.vec3(0.0, -1.0, 0.0)
        self.upward = glm.vec3(0.0, 0.0, 1.0)

    def update(self, yaw, pitch, roll):
        self.forward, self.upward = rotate(self.forward, self.upward, yaw, pitch, roll)
        self.position += self.forward * 0.1
        self.rotation = quat_look_at(self.forward, self.upward)

    def camera(self):
        eye = self.position - self.forward * 4.0 + self.upward * 2.0
        target = self.position + self.forward * 2.0
        up = self.upward
        return zengl.camera(eye, target, up, aspect=window.aspect, fov=45.0)


def render_object(name, position, rotation):
    pipeline = pipelines[name]
    pipeline.uniforms['position'][:] = struct.pack('3f', *position)
    pipeline.uniforms['rotation'][:] = struct.pack('4f', *glm.quat_to_vec4(rotation))
    pipeline.render()


space_ship = SpaceShip()
canisters = np.random.uniform(-50.0, 50.0, (150, 3))


def render():
    window.update()
    ctx.new_frame()
    image.clear()
    depth.clear()
    yaw = -0.01 if window.key_down('KeyA') else 0.01 if window.key_down('KeyD') else 0.0
    pitch = -0.01 if window.key_down('KeyW') else 0.01 if window.key_down('KeyS') else 0.0
    roll = -0.01 if window.key_down('KeyQ') else 0.01 if window.key_down('KeyE') else 0.0
    space_ship.update(yaw, pitch, roll)
    uniform_buffer.write(space_ship.camera())
    render_object('SpaceShip', space_ship.position, space_ship.rotation)
    for c in canisters:
        render_object('Canister', glm.vec3(c), glm.quat(1.0, 0.0, 0.0, 0.0))
    image.blit()
    ctx.end_frame()


if __name__ == '__main__':
    while window.alive():
        render()
