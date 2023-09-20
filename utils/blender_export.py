import struct
import pickle

import bpy

Vertex = struct.Struct('3f3f3f')


objects = []
vertex_data = bytearray()
sprites = bytearray()


def export_image(obj, alpha):
    pixels = obj.data.pixels
    for x in pixels:
        sprites.append(int(min(max(x * alpha * 255.0, 0), 255)))


def export_mesh(obj, ref):
    mesh = obj.data
    mesh.calc_loop_triangles()
    mesh.calc_normals_split()

    rebase_position = ref.matrix_world.inverted() @ obj.matrix_world
    rebase_rotation = rebase_position.to_quaternion()

    buf = bytearray()
    for triangle_loop in mesh.loop_triangles:
        for loop_index in triangle_loop.loops:
            loop = mesh.loops[loop_index]
            x, y, z = rebase_position @ mesh.vertices[loop.vertex_index].co
            r, g, b, _ = mesh.materials[triangle_loop.material_index].node_tree.nodes['Principled BSDF'].inputs[0].default_value
            nx, ny, nz = rebase_rotation @ loop.normal
            buf.extend(Vertex.pack(x, y, z, nx, ny, nz, r, g, b))

    return buf


def export_object(name, obj, ref=None):
    buf = export_mesh(obj, ref or obj)
    vertex_offset = len(vertex_data) // Vertex.size
    vertex_count = len(buf) // Vertex.size
    objects.append((name, vertex_offset, vertex_count))
    vertex_data.extend(buf)


def export_collection(name, collection, ref):
    buf = bytearray()
    for obj in collection.objects:
        buf.extend(export_mesh(obj, ref))
    vertex_offset = len(vertex_data) // Vertex.size
    vertex_count = len(buf) // Vertex.size
    objects.append((name, vertex_offset, vertex_count))
    vertex_data.extend(buf)


export_object('SpaceShip0', bpy.data.objects['craft_speederA'])
export_object('SpaceShip1', bpy.data.objects['craft_speederB'])
export_object('SpaceShip2', bpy.data.objects['craft_speederC'])
export_object('SpaceShip3', bpy.data.objects['craft_speederD'])
export_object('SpaceShip4', bpy.data.objects['craft_cargoA'])
export_object('SpaceShip5', bpy.data.objects['craft_cargoB'])
export_object('SpaceShip6', bpy.data.objects['craft_miner'])
export_object('SpaceShip7', bpy.data.objects['craft_racer'])
export_object('Canister', bpy.data.objects['canister'])
export_object('Smoke', bpy.data.objects['smoke'])

export_collection('Base', bpy.data.collections['base'], bpy.data.objects['platform_large'])
export_collection('Rocket1', bpy.data.collections['rocket-1'], bpy.data.objects['rocket_sidesA'])
export_collection('Rocket2', bpy.data.collections['rocket-2'], bpy.data.objects['rocket_finsB'])

export_image(bpy.data.objects['star'], 1.0)
export_image(bpy.data.objects['nebula'], 0.5)
export_image(bpy.data.objects['planet-1'], 1.0)
export_image(bpy.data.objects['planet-2'], 1.0)
export_image(bpy.data.objects['planet-3'], 1.0)
export_image(bpy.data.objects['planet-4'], 1.0)
export_image(bpy.data.objects['planet-5'], 1.0)


open(bpy.path.abspath('//assets.pickle'), 'wb').write(pickle.dumps({
    'VertexData': bytes(vertex_data),
    'Objects': objects,
    'Sprites': bytes(sprites),
}))