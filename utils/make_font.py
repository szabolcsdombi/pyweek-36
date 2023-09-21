import pickle

from PIL import Image, ImageFont, ImageDraw

img = Image.new('RGBA', (32, 32), '#fff')
draw = ImageDraw.Draw(img)

with open('assets/Inconsolata-Black.ttf', 'rb') as f:
    draw.font = ImageFont.truetype(f, size=24)

pixels = bytearray()
for c in range(32, 127):
    draw.rectangle((0, 0, 32, 32), (0, 0, 0, 0))
    draw.text((1, -2), chr(c), '#fff')
    pixels.extend(img.tobytes('raw', 'RGBA', 0, -1))

with open('assets/font.pickle', 'wb') as f:
    f.write(pickle.dumps({'Font': bytes(pixels)}))
