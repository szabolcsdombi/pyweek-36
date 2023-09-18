import pickle

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

vertex_buffer = ctx.buffer(assets['SpaceShip'])

uniform_buffer = ctx.buffer(size=64)

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        layout (std140) uniform Common {
            mat4 mvp;
        };

        layout (location = 0) in vec3 in_vertex;
        layout (location = 1) in vec3 in_normal;
        layout (location = 2) in vec3 in_color;

        out vec3 v_vertex;
        out vec3 v_normal;
        out vec3 v_color;

        void main() {
            v_vertex = in_vertex;
            v_normal = in_normal;
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
    framebuffer=[image, depth],
    topology='triangles',
    cull_face='back',
    vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 3f', 0, 1, 2),
    vertex_count=vertex_buffer.size // zengl.calcsize('3f 3f 3f'),
)


def render():
    window.update()
    ctx.new_frame()
    image.clear()
    depth.clear()
    camera = zengl.camera((3.0, 2.0, 2.0), (0.0, 0.0, 0.0), aspect=window.aspect, fov=45.0)
    uniform_buffer.write(camera)
    pipeline.render()
    image.blit()
    ctx.end_frame()


if __name__ == '__main__':
    while window.alive():
        render()
