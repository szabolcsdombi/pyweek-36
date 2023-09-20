import math
import pickle
import random
import struct

import glm
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

    def key_pressed(self, key):
        return key in self._wnd.keys and key not in self._wnd.prev_keys

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

uniform_buffer = ctx.buffer(size=96)

sky = ctx.image((128, 128), 'rgba8unorm', array=7, data=assets['Sprites'])

background_vertex_buffer = ctx.buffer(size=1024 * 1024)

background = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        vec3 qtransform(vec4 q, vec3 v) {
            return v + 2.0 * cross(cross(v, q.xyz) - q.w * v, q.xyz);
        }

        layout (std140) uniform Common {
            mat4 mvp;
            vec4 camera_position;
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

        out vec3 v_texcoord;

        void main() {
            vec3 vertex = qtransform(in_rotation, vec3(vertices[gl_VertexID] * 100.0, in_distance * 100.0));
            gl_Position = mvp * vec4(camera_position.xyz + vertex, 1.0);
            gl_Position.w = gl_Position.z;
            v_texcoord = vec3(vertices[gl_VertexID] * 0.5 + 0.5, in_texture);
        }
    ''',
    fragment_shader='''
        #version 330 core

        uniform sampler2DArray Texture;

        in vec3 v_texcoord;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = texture(Texture, v_texcoord);
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
            'image': sky,
        },
    ],
    blend={
        'enable': True,
        'src_color': 'src_alpha',
        'dst_color': 'one_minus_src_alpha',
    },
    framebuffer=[image],
    topology='triangle_strip',
    vertex_buffers=zengl.bind(background_vertex_buffer, '4f 1f 1f /i', 0, 1, 2),
    vertex_count=4,
)

smoke_vertex_buffer = ctx.buffer(assets['Smoke'])
smoke_instance_buffer = ctx.buffer(size=1024 * 1024)
smoke_instances = bytearray()

smoke_pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        vec3 qtransform(vec4 q, vec3 v) {
            return v + 2.0 * cross(cross(v, q.xyz) - q.w * v, q.xyz);
        }

        layout (std140) uniform Common {
            mat4 mvp;
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

        in vec3 v_vertex;
        in vec3 v_normal;

        layout (location = 0) out vec4 out_color;

        void main() {
            vec3 light = vec3(4.0, 3.0, 10.0);
            float lum = dot(normalize(light), normalize(v_normal)) * 0.3 + 0.7;
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
        *zengl.bind(smoke_vertex_buffer, '3f 3f', 0, 1),
        *zengl.bind(smoke_instance_buffer, '3f 4f 1f /i', 2, 3, 4),
    ],
    vertex_count=smoke_vertex_buffer.size // zengl.calcsize('3f 3f'),
)

beam_vertex_buffer = ctx.buffer(assets['Beam'])
beam_instance_buffer = ctx.buffer(size=1024 * 1024)
beam_instances = bytearray()

beam_pipeline = ctx.pipeline(
    vertex_shader='''
        #version 330 core

        vec3 qtransform(vec4 q, vec3 v) {
            return v + 2.0 * cross(cross(v, q.xyz) - q.w * v, q.xyz);
        }

        layout (std140) uniform Common {
            mat4 mvp;
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
        *zengl.bind(beam_vertex_buffer, '3f 1i', 0, 1),
        *zengl.bind(beam_instance_buffer, '3f 4f 1f /i', 2, 3, 4),
    ],
    vertex_count=beam_vertex_buffer.size // zengl.calcsize('3f 1i'),
)


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

            in vec3 v_vertex;
            in vec3 v_normal;
            in vec3 v_color;

            layout (location = 0) out vec4 out_color;

            void main() {
                vec3 light = vec3(4.0, 3.0, 10.0);
                float lum = dot(normalize(light), normalize(v_normal)) * 0.3 + 0.7;
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


pipelines = {
    name: make_pipeline(vertex_offset, vertex_count)
    for name, vertex_offset, vertex_count in assets['Objects']
}


def render_object(name, position, rotation, scale):
    if name == 'Smoke':
        smoke_instances.extend(struct.pack('3f4f1f', *position, *glm.quat_to_vec4(rotation), scale))
        return
    if name == 'Beam':
        beam_instances.extend(struct.pack('3f4f1f', *position, *glm.quat_to_vec4(rotation), scale))
        return
    pipeline = pipelines[name]
    pipeline.uniforms['position'][:] = struct.pack('3f', *position)
    pipeline.uniforms['rotation'][:] = struct.pack('4f', *glm.quat_to_vec4(rotation))
    pipeline.uniforms['scale'][:] = struct.pack('f', scale)
    pipeline.render()


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


class Beam:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.rotation = quat_look_at(velocity, (0.0, 0.0, 1.0))
        self.life = 200
        self.alive = True

    def update(self):
        self.life -= 1
        self.alive = self.life > 0
        self.position += self.velocity

    def render(self):
        render_object('Beam', self.position, self.rotation, 3.0)


class Smoke:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.rotation = random_rotation()
        self.size = random.random() * 0.6 + 0.9
        self.counter = 90
        self.alive = True

    def update(self):
        self.position += self.velocity
        self.velocity *= 0.9
        self.counter -= 1
        self.alive = self.counter > 0

    def render(self):
        render_object('Smoke', self.position, self.rotation, self.size)


class CollectedCanister:
    def __init__(self, collector, canister):
        self.position = canister.position
        self.collector = collector
        self.canister = canister
        self.counter = 10
        self.alive = True

    def update(self):
        self.counter -= 1
        self.alive = self.counter > 0
        f = self.counter / 10.0
        self.canister.update()
        self.position = self.canister.position * f + self.collector.position * (1.0 - f)

    def render(self):
        render_object('Canister', self.position, self.canister.rotation, 1.0)


class SpaceShip:
    def __init__(self, space_ship_model):
        self.space_ship_model = space_ship_model
        self.position = glm.vec3(0.0, 0.0, 0.0)
        self.forward = glm.vec3(0.0, -1.0, 0.0)
        self.upward = glm.vec3(0.0, 0.0, 1.0)
        self.user_input = glm.vec3(0.0, 0.0, 0.0)
        self.yaw_pitch_roll = glm.vec3(0.0, 0.0, 0.0)
        self.rotation = glm.quat(1.0, 0.0, 0.0, 0.0)
        self.alive = True

    def update(self):
        yaw, pitch, roll = self.yaw_pitch_roll
        self.forward, self.upward = rotate(self.forward, self.upward, yaw * 0.5, pitch * 0.5, roll * 0.5)
        temp_forward, temp_upward = rotate(self.forward, self.upward, yaw * 2.0, pitch * 2.0, roll * 2.0)
        self.yaw_pitch_roll = (self.yaw_pitch_roll + self.user_input * 0.01) * 0.9
        self.position += self.forward * 0.3
        self.rotation = quat_look_at(temp_forward, temp_upward)

        for obj in world.nearby(self.position, 4.0):
            if type(obj) is Canister:
                obj.alive = False
                world.add(CollectedCanister(self, obj))

        world.add(Smoke(self.position + self.rotation * glm.vec3(0.35, 0.85, -0.1), self.forward * 0.3 + random_rotation() * glm.vec3(0.01, 0.0, 0.0)))
        world.add(Smoke(self.position + self.rotation * glm.vec3(-0.35, 0.85, -0.1), self.forward * 0.3 + random_rotation() * glm.vec3(0.01, 0.0, 0.0)))

        world.add(Beam(self.position + self.rotation * glm.vec3(0.0, -0.4, -0.1), self.forward * 1.5 + random_rotation() * glm.vec3(0.1, 0.0, 0.0)))

    def render(self):
        render_object(self.space_ship_model, self.position, self.rotation, 1.0)

    def camera(self):
        eye = self.position - self.forward * 6.0 + self.upward * 2.0
        target = self.position + self.forward * 2.0
        up = self.upward
        return zengl.camera(eye, target, up, aspect=window.aspect, fov=60.0)


class WanderingShip:
    def __init__(self):
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

    def update(self):
        user_input = glm.vec3(random.random(), random.random(), random.random()) * 0.08 - 0.04
        self.space_ship.user_input = glm.clamp(self.space_ship.user_input + user_input, glm.vec3(0.0), glm.vec3(1.0))
        self.space_ship.update()
        self.position = self.space_ship.position

    def render(self):
        self.space_ship.render()


class Canister:
    def __init__(self):
        self.position = glm.vec3(random.random(), random.random(), random.random()) * 200.0 - 100.0
        self.rotation = random_rotation()
        self.axis = random_rotation() * glm.vec3(1.0, 0.0, 0.0)
        self.alive = True

    def update(self):
        self.rotation = glm.angleAxis(0.05, self.axis) * self.rotation

    def render(self):
        render_object('Canister', self.position, self.rotation, 1.0)


class SpaceShipControl:
    def __init__(self, space_ship):
        self.space_ship = space_ship

    def update(self):
        yaw = -1.0 if window.key_down('KeyA') else 1.0 if window.key_down('KeyD') else 0.0
        pitch = -1.0 if window.key_down('KeyW') else 1.0 if window.key_down('KeyS') else 0.0
        roll = -1.0 if window.key_down('KeyQ') else 1.0 if window.key_down('KeyE') else 0.0
        self.space_ship.user_input = glm.vec3(yaw, pitch, roll)


class World:
    def __init__(self):
        self.game_objects = []

    def nearby(self, position, radius):
        return [obj for obj in self.game_objects if glm.distance(obj.position, position) <= radius]

    def add(self, obj):
        self.game_objects.append(obj)

    def update(self):
        for obj in self.game_objects:
            obj.update()
        self.game_objects = [obj for obj in self.game_objects if obj.alive]

    def render(self):
        for obj in self.game_objects:
            obj.render()


world = World()
space_ship = SpaceShip('SpaceShip0')
controller = SpaceShipControl(space_ship)


world.add(space_ship)
for _ in range(10):
    world.add(WanderingShip())
for _ in range(150):
    world.add(Canister())


class SmokeParticle:
    def __init__(self, position, velocity, scale, life):
        self.position = position
        self.rotation = random_rotation()
        self.velocity = velocity
        self.scale = scale
        self.life = life

    def render(self, frame):
        position = self.position + self.velocity * math.sqrt(frame * 10.0) / 60.0
        scale = self.scale + frame * 0.01
        if frame < self.life:
            render_object('Smoke', position, self.rotation, scale)


class WindParticle:
    def __init__(self, position, velocity, acceleration, scale, life):
        self.position = position
        self.rotation = random_rotation()
        self.velocity = velocity
        self.acceleration = acceleration
        self.scale = scale
        self.life = life

    def update(self):
        self.position += self.velocity
        self.velocity += self.acceleration
        self.life -= 1

    def render(self):
        render_object('Smoke', self.position, self.rotation, self.scale)


class Explosion:
    def __init__(self, position):
        self.position = position
        self.smoke = []
        self.frame = 0

    def update(self):
        if self.frame == 0:
            for _ in range(60):
                position = random_rotation() * glm.vec3(0.1, 0.0, 0.0)
                velocity = position * 24.0 + random_rotation() * glm.vec3(0.5, 0.0, 0.0)
                scale = random.gauss(3.0, 0.5)
                life = random.randint(20, 30)
                self.smoke.append(SmokeParticle(self.position + position, velocity, scale, life))
        self.frame += 1

    def render(self):
        for s in self.smoke:
            s.render(self.frame)


class Wind:
    def __init__(self):
        self.smoke = []

    def update(self):
        position = rz(random.random() * math.pi * 2.0) * glm.vec3(0.3, 0.0, 0.0)
        velocity = position * 0.02 + random_rotation() * glm.vec3(0.001, 0.0, 0.0)
        scale = random.gauss(1.0, 0.1)
        life = random.randint(20, 30)
        self.smoke.append(WindParticle(position, velocity, velocity * 0.1, scale, life))
        for s in self.smoke:
            s.update()
        self.smoke = [s for s in self.smoke if s.life > 0]

    def render(self):
        for s in self.smoke:
            s.render()


hangar = {
    'time': 0.0,
    'space_ship': 'SpaceShip0',
    'explosion': Explosion((0.0, 0.0, 0.6)),
    'wind': Wind(),
    'playing': False,
}


temp = bytearray()
count = 0

for i in range(1000):
    rotation = random_rotation()
    temp.extend(struct.pack('4f1f1f', *rotation, random.uniform(150.0, 250.0), 0.0)),
    count += 1

for i in range(200):
    rotation = random_rotation()
    temp.extend(struct.pack('4f1f1f', *rotation, 5.0, 1.0)),
    count += 1

for i in range(5):
    rotation = random_rotation()
    temp.extend(struct.pack('4f1f1f', *rotation, random.gauss(25.0, 5.0), i + 2))
    count += 1

background_vertex_buffer.write(temp)
background.instance_count = count


def render():
    window.update()
    ctx.new_frame()
    image.clear()
    depth.clear()
    smoke_instances.clear()
    beam_instances.clear()

    if hangar['playing']:
        controller.update()

        world.update()
        uniform_buffer.write(struct.pack('64s3f4x', space_ship.camera(), *space_ship.position))
        background.render()

        world.render()

    else:
        eye = glm.vec3(0.63, 3.2, 1.36)
        eye = rz((window.mouse[0] - window.size[0] / 2.0) * 0.001) * rx((window.mouse[1] - window.size[1] / 2.0) * 0.001) * eye
        camera = zengl.camera(eye, (0.0, 0.0, 0.2), (0.0, 0.0, 1.0), fov=60.0, aspect=window.aspect)
        uniform_buffer.write(struct.pack('64s3f4x', camera, *eye))
        background.render()

        hangar['explosion'].update()
        hangar['explosion'].render()

        hangar['wind'].update()
        hangar['wind'].render()

        hangar['time'] += 1.0 / 60.0
        for i in range(8):
            if window.key_pressed(f'Key{i + 1}'):
                hangar['space_ship'] = f'SpaceShip{i}'
                hangar['explosion'] = Explosion((0.0, 0.0, 0.6))

        if window.key_pressed('KeyF'):
            space_ship.space_ship_model = hangar['space_ship']
            hangar['playing'] = True

        render_object('Base', glm.vec3(-0.2, -0.2, 0.0), glm.quat(1.0, 0.0, 0.0, 0.0), 1.0)
        render_object(hangar['space_ship'], glm.vec3(0.0, 0.0, 0.6 + math.sin(hangar['time'] * 3.0) * 0.1), rz(math.pi * 0.85), 0.6)

    smoke_instance_buffer.write(smoke_instances)
    smoke_pipeline.instance_count = len(smoke_instances) // zengl.calcsize('3f 4f 1f')
    smoke_pipeline.render()

    beam_instance_buffer.write(beam_instances)
    beam_pipeline.instance_count = len(beam_instances) // zengl.calcsize('3f 4f 1f')
    beam_pipeline.render()

    image.blit()
    ctx.end_frame()


if __name__ == '__main__':
    while window.alive():
        render()
