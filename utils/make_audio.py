import pickle

with open('assets/intro.wav', 'rb') as f:
    intro = f.read()

with open('assets/beauty-flow.wav', 'rb') as f:
    music = f.read()

with open('assets/spaceEngine_002-cut.wav', 'rb') as f:
    engine = f.read()

with open('assets/laserSmall_003.wav', 'rb') as f:
    beam = f.read()

with open('assets/explosionCrunch_001.wav', 'rb') as f:
    explosion = f.read()

with open('assets/forceField_003.wav', 'rb') as f:
    canister = f.read()

assets = {
    'Audio': {
        'Intro': intro,
        'Music': music,
        'Engine': engine,
        'Beam': beam,
        'Explosion': explosion,
        'Canister': canister,
    },
}

with open('assets/audio.pickle', 'wb') as f:
    f.write(pickle.dumps(assets))
