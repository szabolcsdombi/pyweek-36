import argparse
import io
import math
import os
import pickle
import random
import struct
import sys

import glm
import pyglet
import zengl

arg_parser = argparse.ArgumentParser('run_game')
arg_parser.add_argument('--no-audio', action='store_true')
arg_parser.add_argument('--no-fullscreen', action='store_true')
args = arg_parser.parse_args()

pyglet.options['shadow_window'] = False
pyglet.options['debug_gl'] = False

KEYS = {
    pyglet.window.key.A: 'left',
    pyglet.window.key.D: 'right',
    pyglet.window.key.W: 'up',
    pyglet.window.key.S: 'down',
    pyglet.window.key.LEFT: 'left',
    pyglet.window.key.RIGHT: 'right',
    pyglet.window.key.UP: 'up',
    pyglet.window.key.DOWN: 'down',
    pyglet.window.key.Q: 'turn_left',
    pyglet.window.key.E: 'turn_right',
    pyglet.window.key.SPACE: 'space',
    pyglet.window.key.ESCAPE: 'escape',
    pyglet.window.key.BACKSPACE: 'escape',
    pyglet.window.key.LCTRL: 'shoot',
    1: 'shoot',
}


class PygletWindow(pyglet.window.Window):
    def __init__(self):
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
        super().__init__(fullscreen=True, config=config, vsync=True)
        if args.no_fullscreen:
            self.set_fullscreen(False, width=self.width * 9 // 10, height=self.height * 9 // 10)
            self.set_location(self.width // 20, self.height // 20)
        self.set_exclusive_mouse(True)

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
        if key := KEYS.get(button):
            self.keys.add(key)

    def on_mouse_release(self, x, y, button, modifiers):
        if key := KEYS.get(button):
            self.keys.discard(key)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.mouse = (self.mouse[0] + dx, self.mouse[1] + dy)

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse = (self.mouse[0] + dx, self.mouse[1] + dy)

    def on_close(self):
        self.alive = False

    def update(self):
        self.prev_keys = self.keys
        self.keys = self.keys.copy()
        self.mouse = (0, 0)
        return self.alive


class Window:
    def __init__(self):
        self._wnd = PygletWindow()
        self.size = self._wnd.get_framebuffer_size()
        self.aspect = self.size[0] / self.size[1]
        self.loader = None

    @property
    def mouse(self):
        return self._wnd.mouse

    def key_down(self, key):
        return key in self._wnd.keys

    def key_pressed(self, key):
        return key in self._wnd.keys and key not in self._wnd.prev_keys

    def update(self):
        pass

    def alive(self):
        return self._wnd.update()


def load_audio(name):
    return pyglet.media.load(name, io.BytesIO(assets['Audio'][name]), streaming=False)


class Speaker:
    def __init__(self):
        self.intro = load_audio('Intro')
        self.music = load_audio('Music')
        self.hover = load_audio('Engine')
        self.beam = load_audio('Beam')
        self.explosion = load_audio('Explosion')
        self.canister = load_audio('Canister')
        self.players = []

    def reset(self):
        for player in self.players:
            player.pause()
        self.players = []

    def update(self):
        for player in self.players:
            if player.source is None or player.time > player.source.duration:
                player.pause()
        self.players = [player for player in self.players if player.playing]

    def play_intro(self):
        self.players.append(self.intro.play())

    def play_music(self):
        self.players.append(self.music.play())

    def play_hover(self):
        self.players.append(self.hover.play())

    def play_beam(self):
        self.players.append(self.beam.play())

    def play_explosion(self):
        self.players.append(self.explosion.play())

    def play_canister(self):
        self.players.append(self.canister.play())


if not args.no_audio and not sys.platform.startswith('win'):
    try:
        import modernal
        al = modernal.context()

    except:
        exit('Broken pyglet audio detected!\n\nsudo apt-get install libopenal-dev\npip install modernal==0.9.0\n\nor\n\npython3 run_game.py --no-audio')

    class Audio:
        def __init__(self, sources):
            self.sources = sources
            self.idx = 0

        def play(self):
            self.sources[self.idx].change(time=0.0)
            self.sources[self.idx].play()
            self.idx = (self.idx + 1) % len(self.sources)

        def reset(self):
            for s in self.sources:
                s.stop()


    def load_audio(name, sources):
        import wave
        with wave.open(io.BytesIO(assets['Audio'][name])) as w:
            buffer = al.buffer(w.readframes(w.getnframes()))
        return Audio([al.source(buffer) for i in range(sources)])


    class Speaker:
        def __init__(self):
            self.intro = load_audio('Intro', 1)
            self.music = load_audio('Music', 1)
            self.hover = load_audio('Engine', 10)
            self.beam = load_audio('Beam', 50)
            self.explosion = load_audio('Explosion', 50)
            self.canister = load_audio('Canister', 50)

            self.update = lambda: None
            self.play_intro = self.intro.play
            self.play_music = self.music.play
            self.play_hover = self.hover.play
            self.play_beam = self.beam.play
            self.play_explosion = self.explosion.play
            self.play_canister = self.canister.play

        def reset(self):
            self.intro.reset()
            self.music.reset()
            self.hover.reset()
            self.beam.reset()
            self.explosion.reset()
            self.canister.reset()


class NoSpeaker:
    def __init__(self):
        self.reset = lambda: None
        self.update = lambda: None
        self.play_intro = lambda: None
        self.play_music = lambda: None
        self.play_hover = lambda: None
        self.play_beam = lambda: None
        self.play_explosion = lambda: None
        self.play_canister = lambda: None


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


def random_rotation():
    u1 = random.random()
    u2 = random.random()
    u3 = random.random()
    return glm.quat(
        math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2),
        math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2),
        math.sqrt(u1) * math.sin(2.0 * math.pi * u3),
        math.sqrt(u1) * math.cos(2.0 * math.pi * u3),
    )


def rx(angle):
    return glm.quat(math.cos(angle * 0.5), math.sin(angle * 0.5), 0.0, 0.0)


def ry(angle):
    return glm.quat(math.cos(angle * 0.5), 0.0, math.sin(angle * 0.5), 0.0)


def rz(angle):
    return glm.quat(math.cos(angle * 0.5), 0.0, 0.0, math.sin(angle * 0.5))


def smoothstep(edge0, edge1, x):
    t = glm.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def load_score():
    try:
        with open(os.path.join(os.path.dirname(__file__), 'score.txt')) as f:
            return int(f.read().strip())
    except:
        return 0


def save_score(score):
    try:
        with open(os.path.join(os.path.dirname(__file__), 'score.txt'), 'w') as f:
            print(score, file=f)
    except:
        pass


assets_filename = os.path.join(os.path.dirname(__file__), 'assets.pickle')

if not os.path.isfile(assets_filename):
    link = 'https://github.com/szabolcsdombi/pyweek-36/releases/download/2023-09-22/assets.pickle'
    exit(f'Missing assets!\nPlease download the "assets.pickle" and place next to the "run_game.py"\n\n{link}')

with open(assets_filename, 'rb') as f:
    assets = pickle.load(f)

window = Window()

if args.no_audio:
    speaker = NoSpeaker()
else:
    speaker = Speaker()

ctx = zengl.context(window.loader)

image = ctx.image(window.size, 'rgba8unorm', samples=4)
depth = ctx.image(window.size, 'depth24plus', samples=4)
image.clear_value = (0.0, 0.0, 0.0, 1.0)

uniform_buffer = ctx.buffer(size=128)


def set_camera(eye, target, upward):
    forward = glm.normalize(target - eye)
    sideways = glm.normalize(glm.cross(forward, upward))
    upward = glm.normalize(glm.cross(sideways, forward))
    light = glm.normalize(forward * 0.1 + sideways * 2.0 + upward * 10.0)
    camera = zengl.camera(eye, target, upward, fov=60.0, aspect=window.aspect)
    uniform_buffer.write(struct.pack('64s3f4x3f4x', camera, *eye, *light))


def make_object_pipeline(vertex_buffer, vertex_index, vertex_count):
    return ctx.pipeline(
        vertex_shader='''
            #version 330 core

            vec3 qtransform(vec4 q, vec3 v) {
                return v + 2.0 * cross(cross(v, q.xyz) - q.w * v, q.xyz);
            }

            layout (std140) uniform Common {
                mat4 mvp;
                vec4 camera_position;
                vec4 light_direction;
            };

            uniform vec3 position;
            uniform vec4 rotation;
            uniform float scale;

            layout (location = 0) in vec3 in_vertex;
            layout (location = 1) in vec3 in_normal;
            layout (location = 2) in vec3 in_color;

            out vec3 v_vertex;
            out vec3 v_normal;
            out vec3 v_color;

            void main() {
                v_vertex = position + qtransform(rotation, in_vertex * scale);
                v_normal = qtransform(rotation, in_normal);
                v_color = in_color;
                gl_Position = mvp * vec4(v_vertex, 1.0);
            }
        ''',
        fragment_shader='''
            #version 330 core

            layout (std140) uniform Common {
                mat4 mvp;
                vec4 camera_position;
                vec4 light_direction;
            };

            in vec3 v_vertex;
            in vec3 v_normal;
            in vec3 v_color;

            layout (location = 0) out vec4 out_color;

            void main() {
                float lum = dot(light_direction.xyz, normalize(v_normal)) * 0.3 + 0.7;
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
            'scale': 1.0,
        },
        framebuffer=[image, depth],
        topology='triangles',
        cull_face='back',
        vertex_buffers=zengl.bind(vertex_buffer, '3f 3f 3f', 0, 1, 2),
        first_vertex=vertex_index,
        vertex_count=vertex_count,
    )


class ObjectRenderer:
    def __init__(self):
        self.vertex_buffer = ctx.buffer(assets['VertexBuffer']['Objects'])
        self.pipelines = {
            name: make_object_pipeline(self.vertex_buffer, vertex_offset, vertex_count)
            for name, vertex_offset, vertex_count in assets['Objects']
        }

    def render(self, name, position, rotation, scale):
        pipeline = self.pipelines[name]
        pipeline.uniforms['position'][:] = struct.pack('3f', *position)
        pipeline.uniforms['rotation'][:] = struct.pack('4f', *glm.quat_to_vec4(rotation))
        pipeline.uniforms['scale'][:] = struct.pack('f', scale)
        pipeline.render()


class BackgroundRenderer:
    def __init__(self):
        self.instances = bytearray()
        self.instance = struct.Struct('4f 1f 1f 1f')

        self.sprites = ctx.image((128, 128), 'rgba8unorm', array=7, data=assets['Sprites'])
        self.instance_buffer = ctx.buffer(size=1024 * 1024)

        self.pipeline = ctx.pipeline(
            vertex_shader='''
                #version 330 core

                vec3 qtransform(vec4 q, vec3 v) {
                    return v + 2.0 * cross(cross(v, q.xyz) - q.w * v, q.xyz);
                }

                layout (std140) uniform Common {
                    mat4 mvp;
                    vec4 camera_position;
                    vec4 light_direction;
                };

                vec2 vertices[4] = vec2[](
                    vec2(-1.0, -1.0),
                    vec2(-1.0, 1.0),
                    vec2(1.0, -1.0),
                    vec2(1.0, 1.0)
                );

                layout (location = 0) in vec4 in_rotation;
                layout (location = 1) in float in_distance;
                layout (location = 2) in float in_texture;
                layout (location = 3) in float in_alpha;

                out vec3 v_texcoord;
                out float v_alpha;

                void main() {
                    v_alpha = in_alpha;
                    v_texcoord = vec3(vertices[gl_VertexID] * 0.5 + 0.5, in_texture);
                    vec3 vertex = qtransform(in_rotation, vec3(vertices[gl_VertexID] * 100.0, in_distance * 100.0));
                    gl_Position = mvp * vec4(camera_position.xyz + vertex, 1.0);
                    gl_Position.w = gl_Position.z;
                }
            ''',
            fragment_shader='''
                #version 330 core

                uniform sampler2DArray Texture;
                uniform float alpha;

                in vec3 v_texcoord;
                in float v_alpha;

                layout (location = 0) out vec4 out_color;

                void main() {
                    out_color = texture(Texture, v_texcoord);
                    out_color.a *= v_alpha * alpha;
                }
            ''',
            layout=[
                {
                    'name': 'Common',
                    'binding': 0,
                },
                {
                    'name': 'Texture',
                    'binding': 0,
                },
            ],
            resources=[
                {
                    'type': 'uniform_buffer',
                    'binding': 0,
                    'buffer': uniform_buffer,
                },
                {
                    'type': 'sampler',
                    'binding': 0,
                    'image': self.sprites,
                },
            ],
            blend={
                'enable': True,
                'src_color': 'src_alpha',
                'dst_color': 'one_minus_src_alpha',
            },
            uniforms={
                'alpha': 1.0,
            },
            framebuffer=[image],
            topology='triangle_strip',
            vertex_buffers=zengl.bind(self.instance_buffer, '4f 1f 1f 1f /i', 0, 1, 2, 3),
            vertex_count=4,
        )

    def generate(self, planets=True, exclude_home_planet=False):
        self.instances.clear()

        for i in range(1000):
            rotation = random_rotation()
            self.instances.extend(self.instance.pack(*glm.quat_to_vec4(rotation), random.uniform(150.0, 250.0), 0.0, 1.0)),

        for i in range(200):
            rotation = random_rotation()
            self.instances.extend(self.instance.pack(*glm.quat_to_vec4(rotation), 5.0, 1.0, 0.35)),

        if planets:
            for i in range(5):
                if exclude_home_planet and i == 1:
                    continue
                rotation = random_rotation()
                self.instances.extend(self.instance.pack(*glm.quat_to_vec4(rotation), random.gauss(25.0, 5.0), i + 2, 1.0))

        self.instance_buffer.write(self.instances)
        self.pipeline.instance_count = len(self.instances) // self.instance.size
        self.set_alpha(1.0)

    def add_home_planet(self, rotation, distance):
        self.instances.extend(self.instance.pack(*glm.quat_to_vec4(rotation), distance, 3, 1.0))
        self.instance_buffer.write(self.instances)
        self.pipeline.instance_count = len(self.instances) // self.instance.size

    def set_alpha(self, alpha):
        self.pipeline.uniforms['alpha'][:] = struct.pack('f', alpha)

    def render(self):
        self.pipeline.render()


class SmokeRenderer:
    def __init__(self):
        self.instances = bytearray()
        self.instance = struct.Struct('3f 4f 1f')

        self.vertex_buffer = ctx.buffer(assets['VertexBuffer']['Smoke'])
        self.instance_buffer = ctx.buffer(size=1024 * 1024)

        self.pipeline = ctx.pipeline(
            vertex_shader='''
                #version 330 core

                vec3 qtransform(vec4 q, vec3 v) {
                    return v + 2.0 * cross(cross(v, q.xyz) - q.w * v, q.xyz);
                }

                layout (std140) uniform Common {
                    mat4 mvp;
                    vec4 camera_position;
                    vec4 light_direction;
                };

                layout (location = 0) in vec3 in_vertex;
                layout (location = 1) in vec3 in_normal;

                layout (location = 2) in vec3 in_position;
                layout (location = 3) in vec4 in_rotation;
                layout (location = 4) in float in_scale;

                out vec3 v_vertex;
                out vec3 v_normal;

                void main() {
                    v_vertex = in_position + qtransform(in_rotation, in_vertex * in_scale);
                    v_normal = qtransform(in_rotation, in_normal);
                    gl_Position = mvp * vec4(v_vertex, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330 core

                layout (std140) uniform Common {
                    mat4 mvp;
                    vec4 camera_position;
                    vec4 light_direction;
                };

                in vec3 v_vertex;
                in vec3 v_normal;

                layout (location = 0) out vec4 out_color;

                void main() {
                    float lum = dot(light_direction.xyz, normalize(v_normal)) * 0.3 + 0.7;
                    out_color = vec4(lum, lum, lum, 1.0);
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
            vertex_buffers=[
                *zengl.bind(self.vertex_buffer, '3f 3f', 0, 1),
                *zengl.bind(self.instance_buffer, '3f 4f 1f /i', 2, 3, 4),
            ],
            vertex_count=self.vertex_buffer.size // zengl.calcsize('3f 3f'),
        )

    def add(self, position, rotation, scale):
        self.instances.extend(self.instance.pack(*position, *glm.quat_to_vec4(rotation), scale))

    def render(self):
        self.instance_buffer.write(self.instances)
        self.pipeline.instance_count = len(self.instances) // self.instance.size
        self.pipeline.render()
        self.instances.clear()


class BeamRenderer:
    def __init__(self):
        self.instances = bytearray()
        self.instance = struct.Struct('3f 4f 1f')

        self.vertex_buffer = ctx.buffer(assets['VertexBuffer']['Beam'])
        self.instance_buffer = ctx.buffer(size=1024 * 1024)

        self.pipeline = ctx.pipeline(
            vertex_shader='''
                #version 330 core

                vec3 qtransform(vec4 q, vec3 v) {
                    return v + 2.0 * cross(cross(v, q.xyz) - q.w * v, q.xyz);
                }

                layout (std140) uniform Common {
                    mat4 mvp;
                    vec4 camera_position;
                    vec4 light_direction;
                };

                layout (location = 0) in vec3 in_vertex;
                layout (location = 1) in int in_material;

                layout (location = 2) in vec3 in_position;
                layout (location = 3) in vec4 in_rotation;
                layout (location = 4) in float in_scale;

                flat out int v_material;

                void main() {
                    v_material = in_material;
                    vec3 vertex = in_position + qtransform(in_rotation, in_vertex * in_scale);
                    gl_Position = mvp * vec4(vertex, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330 core

                flat in int v_material;

                layout (location = 0) out vec4 out_color;

                void main() {
                    if (v_material == 1) {
                        out_color = vec4(1.0, 0.0, 1.0, 1.0);
                    } else {
                        out_color = vec4(1.0, 1.0, 1.0, 1.0);
                    }
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
            vertex_buffers=[
                *zengl.bind(self.vertex_buffer, '3f 1i', 0, 1),
                *zengl.bind(self.instance_buffer, '3f 4f 1f /i', 2, 3, 4),
            ],
            vertex_count=self.vertex_buffer.size // zengl.calcsize('3f 1i'),
        )

    def add(self, position, rotation, scale):
        self.instances.extend(self.instance.pack(*position, *glm.quat_to_vec4(rotation), scale))

    def render(self):
        self.instance_buffer.write(self.instances)
        self.pipeline.instance_count = len(self.instances) // self.instance.size
        self.pipeline.render()
        self.instances.clear()


class TextRenderer:
    def __init__(self):
        self.instances = bytearray()
        self.instance = struct.Struct('3f')

        self.font_texture = ctx.image((32, 32), 'rgba8unorm', array=95, data=assets['Font'])
        self.instance_buffer = ctx.buffer(size=1024 * 1024)

        self.pipeline = ctx.pipeline(
            vertex_shader='''
                #version 330 core

                #include "screen_size"

                vec2 vertices[4] = vec2[](
                    vec2(0.0, 0.0),
                    vec2(0.0, 1.0),
                    vec2(1.0, 0.0),
                    vec2(1.0, 1.0)
                );

                layout (location = 0) in vec3 in_attributes;

                out vec3 v_texcoord;

                void main() {
                    vec2 position = in_attributes.xy;
                    float texture = in_attributes.z;
                    vec2 vertex = position + vertices[gl_VertexID] * 32.0;
                    gl_Position = vec4(vertex / screen_size * 2.0 - 1.0, 0.0, 1.0);
                    v_texcoord = vec3(vertices[gl_VertexID], texture);
                }
            ''',
            fragment_shader='''
                #version 330 core

                in vec3 v_texcoord;

                uniform sampler2DArray Texture;

                layout (location = 0) out vec4 out_color;

                void main() {
                    out_color = texture(Texture, v_texcoord);
                }
            ''',
            includes={
                'screen_size': f'const vec2 screen_size = vec2({float(window.size[0])}, {float(window.size[1])});',
            },
            layout=[
                {
                    'name': 'Texture',
                    'binding': 0,
                },
            ],
            resources=[
                {
                    'type': 'sampler',
                    'binding': 0,
                    'image': self.font_texture,
                    'wrap_x': 'clamp_to_edge',
                    'wrap_y': 'clamp_to_edge',
                    'min_filter': 'nearest',
                    'mag_filter': 'nearest',
                },
            ],
            blend={
                'enable': True,
                'src_color': 'src_alpha',
                'dst_color': 'one_minus_src_alpha',
            },
            framebuffer=[image],
            topology='triangle_strip',
            vertex_buffers=zengl.bind(self.instance_buffer, '3f /i', 0),
            vertex_count=4,
        )

    def line(self, x, y, text):
        for i in range(len(text)):
            self.instances.extend(self.instance.pack(x + i * 14.0, y, ord(text[i]) - 32))

    def render(self):
        self.instance_buffer.write(self.instances)
        self.pipeline.instance_count = len(self.instances) // self.instance.size
        self.pipeline.render()
        self.instances.clear()


class SpriteRenderer:
    def __init__(self):
        self.instances = bytearray()
        self.instance = struct.Struct('2f 1f 2f 4f')

        self.texture = ctx.image((512, 512), 'rgba8unorm', data=assets['UI']['Texture'])
        self.sprites = assets['UI']['Sprites']

        self.instance_buffer = ctx.buffer(size=1024 * 1024)

        self.pipeline = ctx.pipeline(
            vertex_shader='''
                #version 330 core

                #include "screen_size"

                vec2 vertices[4] = vec2[](
                    vec2(0.0, 0.0),
                    vec2(0.0, 1.0),
                    vec2(1.0, 0.0),
                    vec2(1.0, 1.0)
                );

                layout (location = 0) in vec2 in_position;
                layout (location = 1) in float in_rotation;
                layout (location = 2) in vec2 in_size;
                layout (location = 3) in vec4 in_texcoords;

                out vec2 v_texcoord;

                void main() {
                    mat2 rotation = mat2(cos(in_rotation), -sin(in_rotation), sin(in_rotation), cos(in_rotation));
                    vec2 vertex = in_position + rotation * (in_size * (vertices[gl_VertexID] - 0.5));
                    v_texcoord = mix(in_texcoords.xy, in_texcoords.zw, vertices[gl_VertexID]);
                    gl_Position = vec4(vertex / screen_size * 2.0 - 1.0, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330 core

                uniform sampler2D Texture;

                in vec2 v_texcoord;

                layout (location = 0) out vec4 out_color;

                void main() {
                    out_color = texture(Texture, v_texcoord);
                }
            ''',
            includes={
                'screen_size': f'const vec2 screen_size = vec2({float(window.size[0])}, {float(window.size[1])});',
            },
            layout=[
                {
                    'name': 'Texture',
                    'binding': 0,
                },
            ],
            resources=[
                {
                    'type': 'sampler',
                    'binding': 0,
                    'image': self.texture,
                },
            ],
            blend={
                'enable': True,
                'src_color': 'src_alpha',
                'dst_color': 'one_minus_src_alpha',
            },
            framebuffer=[image],
            topology='triangle_strip',
            vertex_buffers=zengl.bind(self.instance_buffer, '2f 1f 2f 4f /i', 0, 1, 2, 3),
            vertex_count=4,
        )

    def add(self, name, position, rotation):
        size, texcoords = self.sprites[name]
        self.instances.extend(self.instance.pack(*position, rotation, *size, *texcoords))

    def render(self):
        self.instance_buffer.write(self.instances)
        self.pipeline.instance_count = len(self.instances) // self.instance.size
        self.pipeline.render()
        self.instances.clear()


object_renderer = ObjectRenderer()
background_renderer = BackgroundRenderer()
smoke_renderer = SmokeRenderer()
beam_renderer = BeamRenderer()
text_renderer = TextRenderer()
sprite_renderer = SpriteRenderer()


class Beam:
    def __init__(self, owner, position, velocity):
        self.owner = owner
        self.position = position
        self.velocity = velocity
        self.rotation = quat_look_at(velocity, (0.0, 0.0, 1.0))
        self.life = 200
        self.alive = True

    def update(self, world):
        self.life -= 1
        self.alive = self.alive and self.life > 0
        self.position += self.velocity

    def render(self):
        beam_renderer.add(self.position, self.rotation, 3.0)


class ShipSmoke:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.rotation = random_rotation()
        self.size = random.random() * 0.6 + 0.9
        self.counter = 90
        self.alive = True

    def update(self, world):
        self.position += self.velocity
        self.velocity *= 0.9
        self.counter -= 1
        self.alive = self.counter > 0

    def render(self):
        smoke_renderer.add(self.position, self.rotation, self.size)


class CollectedCanister:
    def __init__(self, collector, canister):
        self.position = canister.position
        self.collector = collector
        self.canister = canister
        self.counter = 10
        self.alive = True

    def update(self, world):
        self.counter -= 1
        self.alive = self.counter > 0
        f = self.counter / 10.0
        self.canister.update(world)
        self.position = self.canister.position * f + self.collector.position * (1.0 - f)

    def render(self):
        object_renderer.render('Canister', self.position, self.canister.rotation, 1.0)


class SpaceShip:
    def __init__(self, space_ship_model):
        self.space_ship_model = space_ship_model
        self.position = glm.vec3(0.0, 0.0, 0.0)
        self.forward = glm.vec3(0.0, -1.0, 0.0)
        self.upward = glm.vec3(0.0, 0.0, 1.0)
        self.user_input = glm.vec3(0.0, 0.0, 0.0)
        self.yaw_pitch_roll = glm.vec3(0.0, 0.0, 0.0)
        self.rotation = glm.quat(1.0, 0.0, 0.0, 0.0)
        self.canisters_collected = 0
        self.shooting = False
        self.alive = True
        self.frame = 0

        perks = {
            'SpaceShip0': {'speed': 0.3, 'mobility': 1.0, 'health': 10, 'fire_cooldown': 5, 'fire_spread': 0.1, 'dual_cannon': False, 'smoke_offset': (0.35, 0.1)},
            'SpaceShip1': {'speed': 0.3, 'mobility': 1.0, 'health': 10, 'fire_cooldown': 3, 'fire_spread': 0.08, 'dual_cannon': False, 'smoke_offset': (0.4, 0.05)},
            'SpaceShip2': {'speed': 0.4, 'mobility': 1.2, 'health': 10, 'fire_cooldown': 3, 'fire_spread': 0.08, 'dual_cannon': False, 'smoke_offset': (0.4, 0.0)},
            'SpaceShip3': {'speed': 0.4, 'mobility': 1.2, 'health': 10, 'fire_cooldown': 3, 'fire_spread': 0.06, 'dual_cannon': False, 'smoke_offset': (0.4, 0.0)},
            'SpaceShip4': {'speed': 0.5, 'mobility': 1.0, 'health': 20, 'fire_cooldown': 3, 'fire_spread': 0.06, 'dual_cannon': False, 'smoke_offset': (0.6, 0.1)},
            'SpaceShip5': {'speed': 0.5, 'mobility': 1.2, 'health': 30, 'fire_cooldown': 3, 'fire_spread': 0.03, 'dual_cannon': False, 'smoke_offset': (0.5, 0.05)},
            'SpaceShip6': {'speed': 0.5, 'mobility': 1.2, 'health': 20, 'fire_cooldown': 3, 'fire_spread': 0.03, 'dual_cannon': True, 'smoke_offset': (0.4, 0.0)},
            'SpaceShip7': {'speed': 0.8, 'mobility': 2.0, 'health': 20, 'fire_cooldown': 3, 'fire_spread': 0.001, 'dual_cannon': False, 'smoke_offset': (0.3, 0.1)},
        }[space_ship_model]

        self.speed = perks['speed']
        self.mobility = perks['mobility']
        self.health = perks['health']
        self.fire_cooldown = perks['fire_cooldown']
        self.fire_spread = perks['fire_spread']
        self.dual_cannon = perks['dual_cannon']
        self.smoke_offset = perks['smoke_offset']

    def update(self, world):
        self.frame += 1
        self.user_input = glm.clamp(self.user_input, glm.vec3(-self.mobility), glm.vec3(self.mobility))
        yaw, pitch, roll = self.yaw_pitch_roll
        self.forward, self.upward = rotate(self.forward, self.upward, yaw * 0.5, pitch * 0.5, roll * 0.5)
        temp_forward, temp_upward = rotate(self.forward, self.upward, yaw * 2.0, pitch * 2.0, roll * 2.0)
        self.yaw_pitch_roll = (self.yaw_pitch_roll + self.user_input * 0.01) * 0.9
        self.position += self.forward * self.speed
        self.rotation = quat_look_at(temp_forward, temp_upward)

        if self.health < 0:
            world.add(ExplosionChain(self.position, self.forward * 0.2))
            speaker.play_explosion()
            self.alive = False

        for obj in world.nearby(self.position, 4.0):
            if type(obj) is Canister:
                world.add(CollectedCanister(self, obj))
                if glm.distance(self.position, world.listener) < 40.0:
                    speaker.play_canister()
                self.canisters_collected += 1
                obj.alive = False

            if type(obj) is Beam:
                if obj.owner is not self:
                    obj.alive = False
                    self.health -= 1

            if type(obj) is WanderingShip and obj.space_ship is not self:
                obj.space_ship.alive = False
                world.add(ExplosionChain(obj.space_ship.position, obj.space_ship.forward * 0.2))
                world.add(ExplosionChain(self.position, self.forward * 0.2))
                speaker.play_explosion()
                self.alive = False

        s0, s1 = self.smoke_offset
        world.add(ShipSmoke(self.position + self.rotation * glm.vec3(s0, 0.85, -s1), self.forward * 0.3 + random_rotation() * glm.vec3(0.01, 0.0, 0.0)))
        world.add(ShipSmoke(self.position + self.rotation * glm.vec3(-s0, 0.85, -s1), self.forward * 0.3 + random_rotation() * glm.vec3(0.01, 0.0, 0.0)))

        if self.shooting and self.frame % self.fire_cooldown == 0:
            if self.dual_cannon:
                world.add(Beam(self, self.position + self.rotation * glm.vec3(0.5, -0.4, -0.1), self.forward * 1.5 + random_rotation() * glm.vec3(self.fire_spread, 0.0, 0.0)))
                world.add(Beam(self, self.position + self.rotation * glm.vec3(-0.5, -0.4, -0.1), self.forward * 1.5 + random_rotation() * glm.vec3(self.fire_spread, 0.0, 0.0)))
            else:
                world.add(Beam(self, self.position + self.rotation * glm.vec3(0.0, -0.4, -0.1), self.forward * 1.5 + random_rotation() * glm.vec3(self.fire_spread, 0.0, 0.0)))
            if glm.distance(self.position, world.listener) < 40.0:
                speaker.play_beam()

    def render(self):
        object_renderer.render(self.space_ship_model, self.position, self.rotation, 1.0)

    def camera(self):
        eye = self.position - self.forward * 6.0 + self.upward * 2.0
        target = self.position + self.forward * 2.0
        upward = self.upward
        return eye, target, upward


class WanderingShip:
    def __init__(self):
        self.cooldown = random.randrange(180, 240)
        self.space_ship = SpaceShip(f'SpaceShip{random.randrange(0, 8)}')
        position = random_rotation() * glm.vec3(100.0, 0.0, 0.0)
        forward = glm.normalize(random_rotation() * glm.vec3(20.0, 0.0, 0.0) - position)
        upward = random_rotation() * glm.vec3(1.0, 0.0, 0.0)
        sideways = glm.normalize(glm.cross(forward, upward))
        upward = glm.normalize(glm.cross(sideways, forward))
        self.space_ship.position = position
        self.space_ship.forward = forward
        self.position = self.space_ship.position
        self.alive = True

    def update(self, world):
        self.cooldown -= 1
        if self.cooldown < 0:
            self.space_ship.shooting = not self.space_ship.shooting
            self.cooldown = random.randrange(60, 60 + (60 if self.space_ship.shooting else 180))
        user_input = glm.vec3(random.random(), random.random(), random.random()) * 0.08 - 0.04
        self.space_ship.user_input = glm.clamp(self.space_ship.user_input + user_input, glm.vec3(-1.0), glm.vec3(1.0))
        pdist = glm.length(self.space_ship.position)
        if pdist > 100.0:
            correction = -self.space_ship.position / pdist
            factor = glm.clamp((pdist - 100.0) * 0.003, 0.0, 0.05)
            self.space_ship.forward = glm.normalize(self.space_ship.forward + correction * factor)
            sideways = glm.normalize(glm.cross(self.space_ship.forward, self.space_ship.upward))
            self.space_ship.upward = glm.normalize(glm.cross(sideways, self.space_ship.forward))
            self.space_ship.user_input *= 0.95
        self.space_ship.update(world)
        self.alive = self.space_ship.alive
        self.position = self.space_ship.position

    def render(self):
        self.space_ship.render()


class Canister:
    def __init__(self):
        self.position = random_rotation() * glm.vec3(random.uniform(30.0, 100.0))
        self.rotation = random_rotation()
        self.axis = random_rotation() * glm.vec3(1.0, 0.0, 0.0)
        self.alive = True

    def update(self, world):
        self.rotation = glm.angleAxis(0.05, self.axis) * self.rotation

    def render(self):
        object_renderer.render('Canister', self.position, self.rotation, 1.0)


class SpaceShipControl:
    def __init__(self, space_ship):
        self.space_ship = space_ship

    def update(self):
        yaw = -1.0 if window.key_down('left') else 1.0 if window.key_down('right') else 0.0
        pitch = -1.0 if window.key_down('up') else 1.0 if window.key_down('down') else 0.0
        roll = -1.0 if window.key_down('turn_left') else 1.0 if window.key_down('turn_right') else 0.0
        yaw += window.mouse[0] / 50.0
        pitch += window.mouse[1] / 50.0
        self.space_ship.user_input = glm.vec3(yaw, pitch, roll)
        self.space_ship.shooting = window.key_down('shoot')


class World:
    def __init__(self):
        self.game_objects = []
        self.listener = glm.vec3(0.0, 0.0, 0.0)

    def nearby(self, position, radius):
        return [obj for obj in self.game_objects if glm.distance(obj.position, position) <= radius]

    def add(self, obj):
        self.game_objects.append(obj)

    def update(self, listener):
        self.listener = listener
        for obj in self.game_objects:
            obj.update(self)
        self.game_objects = [obj for obj in self.game_objects if obj.alive]

    def render(self):
        for obj in self.game_objects:
            obj.render()


class SmokeParticle:
    def __init__(self, position, velocity, scale, life):
        self.position = position
        self.rotation = random_rotation()
        self.velocity = velocity
        self.scale = scale
        self.life = life
        self.alive = True
        self.frame = 0

    def update(self, world):
        self.frame += 1
        self.life -= 1
        self.alive = self.life > 0

    def render(self):
        position = self.position + self.velocity * math.sqrt(self.frame * 10.0) / 60.0
        scale = self.scale + self.frame * 0.01
        smoke_renderer.add(position, self.rotation, scale)


class WindParticle:
    def __init__(self, position, velocity, acceleration, scale, life):
        self.position = position
        self.rotation = random_rotation()
        self.velocity = velocity
        self.acceleration = acceleration
        self.scale = scale
        self.life = life
        self.alive = True

    def update(self, world):
        self.position += self.velocity
        self.velocity += self.acceleration
        self.life -= 1
        self.alive = self.life > 0

    def render(self):
        smoke_renderer.add(self.position, self.rotation, self.scale)


class Explosion:
    def __init__(self, position, life):
        self.position = position
        self.life = life
        self.alive = False

    def update(self, world):
        for _ in range(60):
            position = random_rotation() * glm.vec3(0.1, 0.0, 0.0)
            velocity = position * 24.0 + random_rotation() * glm.vec3(0.5, 0.0, 0.0)
            scale = random.gauss(3.0, 0.5)
            life = random.randint(self.life, self.life + 10)
            world.add(SmokeParticle(self.position + position, velocity, scale, life))

        self.alive = False

    def render(self):
        pass


class ExplosionChain:
    def __init__(self, position, forward):
        self.position = position
        self.forward = forward
        self.countdown = 30
        self.alive = True

    def update(self, world):
        if self.countdown % 5 == 0:
            world.add(Explosion(self.position, 60))
        self.position += self.forward
        self.countdown -= 1
        self.alive = self.countdown > 0

    def render(self):
        pass


class Wind:
    def __init__(self):
        self.position = glm.vec3(0.0, 0.0, 0.0)
        self.alive = True

    def update(self, world):
        position = rz(random.random() * math.pi * 2.0) * glm.vec3(0.25, 0.0, 0.0)
        velocity = position * 0.02 + random_rotation() * glm.vec3(0.001, 0.0, 0.0)
        scale = random.gauss(1.0, 0.1)
        life = random.randint(20, 30)
        world.add(WindParticle(position, velocity, velocity * 0.1, scale, life))

    def render(self):
        pass


class Intro:
    def __init__(self):
        self.frame = 0
        background_renderer.generate(planets=False)
        background_renderer.add_home_planet(ry(0.225), 15.0)
        self.view = glm.vec3(1.0, 0.0, 0.0)

    def render(self):
        self.frame += 1

        rotation_speed = 0.0005 * (1.0 - smoothstep(2000.0, 2120.0, self.frame))
        self.view = ry(-rotation_speed) * self.view
        background_renderer.set_alpha(min(max((self.frame - 150.0) / 120.0, 0.0), 1.0))

        if window.key_pressed('space') or window.key_pressed('escape'):
            g.scene = Base('SpaceShip0')

        lines = [
            (30, 'In the year 3077,'),
            (150, 'the Milky Way Galaxy is in the midst of an energy crisis.'),
            (370, ''),
            (370, 'The primary source of energy, a rare crystalline element called "Dark Matter" is nearing depletion.'),
            (640, 'Dark Matter is primarily stored in canisters that have been scattered throughout space over centuries'),
            (840, 'due to space wars, trading routes, and exploration mishaps.'),
            (1110, ''),
            (1110, 'Captain Neil Starbreaker is the fearless pilot of the spacecraft "Nebula Harvester".'),
            (1340, 'Neil used to be a space pirate but has since reformed after witnessing'),
            (1570, 'the dire effects of the energy crisis on his home planet, Noverra.'),
            (1800, ''),
            (1800, 'Join Captain Starbreaker on the "Nebula Harvester" and help save the galaxy.'),
        ]

        if self.frame == 30:
            speaker.play_intro()

        show = [text for when, text in lines if self.frame > when]

        for i, text in enumerate(show):
            text_renderer.line(100, 180 - (i - len(show)) * 30, text)

        if self.frame > 2000:
            text_renderer.line(window.size[0] / 2 - 180, 100, 'press [SPACE] to continue')

        set_camera(glm.vec3(0.0, 0.0, 0.0), self.view, glm.cross(self.view, (0.0, 1.0, 0.0)))
        background_renderer.render()
        text_renderer.render()


class Outro:
    def __init__(self):
        self.frame = 0

    def render(self):
        self.frame += 1
        background_renderer.set_alpha(min(max(1.0 - (self.frame - 0.0) / 30.0, 0.0), 1.0))
        background_renderer.render()
        if self.frame == 60:
            exit()


class Base:
    def __init__(self, space_ship_model):
        self.unlocked_ships = g.total_score // 20 + 1
        self.ships = [
            'SpaceShip0', 'SpaceShip1', 'SpaceShip2', 'SpaceShip3',
            'SpaceShip4', 'SpaceShip5', 'SpaceShip6', 'SpaceShip7',
        ][:self.unlocked_ships]
        if self.unlocked_ships >= 8:
            self.unlocked_ships = 'all'

        self.world = World()
        self.world.add(Wind())
        speaker.reset()
        background_renderer.generate(exclude_home_planet=True)
        self.space_ship_model = space_ship_model
        self.view = (0.0, 0.0)
        self.frame = 0
        self.leaving = False

    def render(self):
        speaker.update()
        if self.frame % 100 == 0:
            speaker.play_hover()
        self.view = (self.view[0] + window.mouse[0] * 0.001, self.view[1] + window.mouse[1] * 0.001)
        self.view = min(max(self.view[0], -1.0), 1.0), min(max(self.view[1], -0.5), 0.5)
        self.frame += 1

        if self.leaving:
            text_renderer.line(window.size[0] / 2.0 - 180, window.size[1] / 2.0 + 120, 'Exit the game?')
            text_renderer.line(window.size[0] / 2.0 - 180, window.size[1] / 2.0 + 70, 'press [SPACE] to exit')
            text_renderer.line(window.size[0] / 2.0 - 180, window.size[1] / 2.0 + 40, 'press [ESCAPE] to continue')
            if window.key_pressed('space'):
                g.scene = Outro()
        else:
            text_renderer.line(window.size[0] / 2 - 180, 100, 'press [SPACE] to start')

        if window.key_pressed('escape'):
            self.leaving = not self.leaving

        self.world.update(glm.vec3(0.0, 0.0, 0.0))

        eye = glm.vec3(0.63, 3.2, 1.36)
        eye = rz(self.view[0]) * rx(self.view[1]) * eye
        eye.z = max(eye.z, 0.1)
        set_camera(eye, glm.vec3(0.0, 0.0, 0.2), glm.vec3(0.0, 0.0, 1.0))
        background_renderer.render()
        self.world.render()

        if not self.leaving:
            if window.key_pressed('left'):
                self.space_ship_model = self.ships[(self.ships.index(self.space_ship_model) - 1) % len(self.ships)]
                self.world.add(Explosion((0.0, 0.0, 0.6), 20))
                speaker.play_explosion()

            if window.key_pressed('right'):
                self.space_ship_model = self.ships[(self.ships.index(self.space_ship_model) + 1) % len(self.ships)]
                speaker.play_explosion()
                self.world.add(Explosion((0.0, 0.0, 0.6), 20))

            if window.key_pressed('space'):
                if g.total_score > 0:
                    g.scene = Play(self.space_ship_model)
                else:
                    g.scene = Tutorial()

            object_renderer.render('Base', glm.vec3(-0.2, -0.2, 0.0), glm.quat(1.0, 0.0, 0.0, 0.0), 1.0)
            object_renderer.render(self.space_ship_model, glm.vec3(0.0, 0.0, 0.6 + math.sin(self.frame / 20.0) * 0.1), rz(math.pi * 0.85), 0.6)

        if self.leaving:
            smoke_renderer.instances.clear()

        text_renderer.line(window.size[0] - 300, window.size[1] - 50, f'Dark Matter: {g.total_score}')
        text_renderer.line(window.size[0] - 300, window.size[1] - 80, f'Unlocked Ships: {self.unlocked_ships}')

        smoke_renderer.render()
        beam_renderer.render()
        text_renderer.render()


def render_minimap(space_ship, world):
    x, y = window.size[0] - 150, 150
    sprite_renderer.add('Minimap', (x, y), 0.0)
    p = space_ship.position
    r = glm.inverse(space_ship.rotation)
    for obj in world.game_objects:
        if type(obj) is Canister:
            t = r * (obj.position - p) / 100.0
            if abs(t.z) > 0.1:
                continue
            t.z = 0.0
            if glm.length(t) > 1.0:
                t = glm.normalize(t)
            sprite_renderer.add('Canister', (x - t.x * 100.0, y - t.y * 100.0), 0.0)
        if type(obj) is SpaceShip and obj is not space_ship:
            t = r * (obj.position - p) / 100.0
            t.z = 0.0
            if glm.length(t) > 1.0:
                t = glm.normalize(t)
            q = r * obj.rotation * glm.vec3(0.0, -1.0, 0.0)
            e = math.atan2(q.x, q.y) + math.pi
            sprite_renderer.add('SpaceShip2', (x - t.x * 100.0, y - t.y * 100.0), e)
    sprite_renderer.add('SpaceShip1', (x, y), 0.0)
    sprite_renderer.add('MinimapBorder', (x, y), 0.0)


class Play:
    def __init__(self, space_ship_model):
        self.world = World()
        self.space_ship = SpaceShip(space_ship_model)
        self.controller = SpaceShipControl(self.space_ship)
        background_renderer.generate()
        speaker.reset()
        speaker.play_music()
        self.frame = 0
        self.leaving = False

        self.world.add(self.space_ship)
        for _ in range(10):
            self.world.add(WanderingShip())
        for _ in range(150):
            self.world.add(Canister())

    def render(self):
        speaker.update()
        if self.frame % 100 == 0:
            speaker.play_hover()
        self.frame += 1

        if not self.leaving and not self.space_ship.alive:
            text_renderer.line(window.size[0] / 2.0 - 180, window.size[1] / 2.0 + 120, 'Game over')
            text_renderer.line(window.size[0] / 2.0 - 180, window.size[1] / 2.0 + 70, 'press [SPACE] to exit')
            if window.key_pressed('space'):
                g.total_score += self.space_ship.canisters_collected
                save_score(g.total_score)
                g.scene = Base(self.space_ship.space_ship_model)

        if self.leaving:
            text_renderer.line(window.size[0] / 2.0 - 180, window.size[1] / 2.0 + 120, 'Abort mission?')
            text_renderer.line(window.size[0] / 2.0 - 180, window.size[1] / 2.0 + 70, 'press [SPACE] to exit')
            text_renderer.line(window.size[0] / 2.0 - 180, window.size[1] / 2.0 + 40, 'press [ESCAPE] to continue')
            if window.key_pressed('space'):
                g.total_score += self.space_ship.canisters_collected
                save_score(g.total_score)
                g.scene = Base(self.space_ship.space_ship_model)

        if window.key_pressed('escape'):
            self.leaving = not self.leaving

        self.controller.update()

        self.world.update(self.space_ship.position)
        eye, target, upward = self.space_ship.camera()
        set_camera(eye, target, upward)
        background_renderer.render()

        self.world.render()

        smoke_renderer.render()
        beam_renderer.render()

        render_minimap(self.space_ship, self.world)
        sprite_renderer.render()

        text_renderer.line(window.size[0] - 250, window.size[1] - 50, f'Score: {self.space_ship.canisters_collected}')
        text_renderer.line(window.size[0] - 250, window.size[1] - 80, f'Canisters: {sum(1 for x in self.world.game_objects if type(x) is Canister)}')
        text_renderer.line(window.size[0] - 250, window.size[1] - 110, f'Rivals: {sum(1 for x in self.world.game_objects if type(x) is WanderingShip)}')
        text_renderer.render()


class Tutorial:
    def __init__(self):
        self.world = World()
        self.space_ship = SpaceShip('SpaceShip0')
        self.controller = SpaceShipControl(self.space_ship)
        background_renderer.generate(planets=False)
        speaker.reset()
        speaker.play_music()
        self.world.add(self.space_ship)
        self.frame = 0
        self.leaving = False
        self.hint = 0

    def render(self):
        speaker.update()
        if self.frame % 100 == 0:
            speaker.play_hover()
        self.frame += 1

        cx, cy = window.size[0] / 2.0, window.size[1] / 2.0

        if self.leaving:
            text_renderer.line(cx - 180, cy + 120, 'Abort mission?')
            text_renderer.line(cx - 180, cy + 70, 'press [SPACE] to exit')
            text_renderer.line(cx - 180, cy + 40, 'press [ESCAPE] to continue')
            if window.key_pressed('space'):
                g.total_score += self.space_ship.canisters_collected
                save_score(g.total_score)
                g.scene = Base(self.space_ship.space_ship_model)

        elif self.hint == 0:
            text_renderer.line(cx - 220, cy + 100 + 220, 'Welcome to the Tutorial')
            text_renderer.line(cx - 220, cy + 100 + 130, 'For the next hint press [SPACE]')
            text_renderer.line(cx - 220, cy + 100 + 100, 'To quit anytime press [ESCAPE]')

        elif self.hint == 1:
            text_renderer.line(cx - 220, cy + 100 + 220, 'You can now control the ship')
            text_renderer.line(cx - 220, cy + 100 + 130, 'Use the mouse for smooth navigation')
            text_renderer.line(cx - 220, cy + 100 + 100, 'Click to shoot laser beams')

        elif self.hint == 2:
            text_renderer.line(cx - 300, cy + 100 + 220, 'Your stats are tracked up top')
            text_renderer.line(cx - 300, cy + 100 + 130, 'The goal is to collect canisters of Dark Matter')
            text_renderer.line(cx - 300, cy + 100 + 100, 'For every 20 canisters a new Space Ship is unlocked')

        elif self.hint == 3:
            text_renderer.line(cx - 220, cy + 100 + 220, 'There is a minimap in the corner')
            text_renderer.line(cx - 220, cy + 100 + 130, 'The minimap shows the nearby canisters')
            text_renderer.line(cx - 220, cy + 100 + 100, 'Collect at least one canister')

        elif self.hint == 4:
            pass

        elif self.hint == 5:
            text_renderer.line(cx - 300, cy + 100 + 220, 'Don\'t you think I give you all these canisters for free?')
            text_renderer.line(cx - 300, cy + 100 + 130, 'Your rivals also need this valuable Dark Matter')
            text_renderer.line(cx - 300, cy + 100 + 100, 'You can shoot them down you know')

        elif self.hint == 6:
            pass

        elif self.hint == 7:
            text_renderer.line(cx - 180, cy + 180, 'Congratulations!')
            text_renderer.line(cx - 180, cy + 130, 'You have completed the tutorial')
            text_renderer.line(cx - 180, cy + 40, 'press [SPACE] to continue')

        if window.key_pressed('space') and self.hint != 4 and self.hint != 6:
            if self.hint == 3:
                for _ in range(150):
                    canister = Canister()
                    canister.position += self.space_ship.position
                    self.world.add(canister)
            if self.hint == 5:
                space_ship = SpaceShip('SpaceShip3')
                sideways = glm.cross(self.space_ship.forward, self.space_ship.upward)
                space_ship.position = self.space_ship.position + self.space_ship.forward * 60.0 - sideways * 50.0
                space_ship.upward = self.space_ship.upward
                space_ship.forward = sideways
                space_ship.user_input.z = 1.0
                self.world.add(space_ship)
                for i in range(15):
                    canister = Canister()
                    canister.position = space_ship.position + space_ship.forward * (30.0 + i * 3.0)
                    self.world.add(canister)
            if self.hint == 7:
                g.total_score += self.space_ship.canisters_collected
                save_score(g.total_score)
                g.scene = Base(self.space_ship.space_ship_model)

            self.hint += 1

        if window.key_pressed('escape'):
            self.leaving = not self.leaving

        if self.hint >= 1 and self.hint != 5:
            self.controller.update()

        self.world.update(self.space_ship.position)

        canisters = sum(1 for x in self.world.game_objects if type(x) is Canister)
        rivals = sum(1 for x in self.world.game_objects if type(x) is SpaceShip and x is not self.space_ship)

        if self.hint == 4 and canisters < 150:
            for obj in self.world.game_objects:
                if type(obj) is Canister:
                    obj.alive = False
                self.space_ship.user_input = glm.vec3(0.0, 0.0, 0.0)
            self.hint += 1

        if self.hint == 6 and (canisters == 0 or rivals == 0):
            self.hint += 1

        eye, target, upward = self.space_ship.camera()
        set_camera(eye, target, upward)
        background_renderer.render()

        self.world.render()

        smoke_renderer.render()
        beam_renderer.render()

        if self.hint >= 2:
            text_renderer.line(window.size[0] - 250, window.size[1] - 50, f'Score: {self.space_ship.canisters_collected}')
            text_renderer.line(window.size[0] - 250, window.size[1] - 80, f'Canisters: {canisters}')
            text_renderer.line(window.size[0] - 250, window.size[1] - 110, f'Rivals: {rivals}')

        if self.hint >= 3:
            render_minimap(self.space_ship, self.world)

        sprite_renderer.render()
        text_renderer.render()


class g:
    scene = Intro()
    total_score = load_score()


def render():
    window.update()
    ctx.new_frame()
    image.clear()
    depth.clear()

    g.scene.render()

    image.blit()
    ctx.end_frame()


if __name__ == '__main__':
    @window._wnd.event
    def on_draw():
        render()
        if not window.alive():
            exit()
    pyglet.app.run()
