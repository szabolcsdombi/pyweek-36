import pickle

from PIL import Image

buf = bytearray()


def export(path):
    img = Image.open(path)
    img = img.convert('RGBA').transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    img = img.resize((128, 128), Image.Resampling.LANCZOS)
    buf.extend(img.tobytes())


img = Image.open(f'assets/Custom/star.png')
img = img.convert('RGBA').transpose(Image.Transpose.FLIP_TOP_BOTTOM)
buf.extend(img.tobytes())

img = Image.open(f'assets/Custom/clouds.png')
img = img.convert('RGBA').transpose(Image.Transpose.FLIP_TOP_BOTTOM)
buf.extend(img.tobytes())

export('assets/Planets/planet03.png')
export('assets/Planets/planet05.png')
export('assets/Planets/planet06.png')
export('assets/Planets/planet08.png')
export('assets/Planets/planet09.png')

with open('assets/sprites.pickle', 'wb') as f:
    f.write(pickle.dumps({'Sprites': bytes(buf)}))
