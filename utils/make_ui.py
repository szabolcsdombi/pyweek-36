import os
import pickle
import struct
import zipfile

from PIL import Image, ImageDraw

SIZE = 512

def add(x, y, name, path, transpose=None):
    sprite = Image.open(path).convert('RGBA')
    if transpose:
        sprite = sprite.transpose(transpose)
    w, h = sprite.size
    img.paste(sprite, (x, SIZE - (y + h), x + w, SIZE - y))
    lookup[name] = ((w, h), ((x - 0.5) / SIZE, (y - 0.5) / SIZE, (x + w + 0.5) / SIZE, (y + h + 0.5) / SIZE))

lookup = {}
img = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))

add(480, 10, 'SpaceShip1', 'assets/UI/SpaceShip1.png')
add(480, 60, 'SpaceShip2', 'assets/UI/SpaceShip2.png')
add(480, 100, 'Canister', 'assets/UI/Canister.png')
add(2, 2, 'Minimap', 'assets/UI/Minimap.png')
add(236, 2, 'MinimapBorder', 'assets/UI/MinimapBorder.png')

with open('assets/ui.pickle', 'wb') as f:
    f.write(pickle.dumps({
        'UI': {
            'Texture': img.transpose(Image.Transpose.FLIP_TOP_BOTTOM).tobytes(),
            'Sprites': lookup,
        },
    }))

draw = ImageDraw.Draw(img)
for name in lookup:
    x1, y1, x2, y2 = lookup[name][1]
    draw.rectangle((x1 * SIZE, SIZE - y2 * SIZE, x2 * SIZE, SIZE - y1 * SIZE), None, '#ff0')
    draw.text((x1 * SIZE, SIZE - y2 * SIZE - 10), name, '#f00')

img.save('assets/ui_debug.png')
