import pickle

with open('assets/Music/impact-prelude.wav', 'rb') as f:
    intro = f.read()

with open('assets/Music/beauty-flow.wav', 'rb') as f:
    music = f.read()

with open('assets/SciFiSounds/spaceEngine_002.wav', 'rb') as f:
    engine = f.read()

with open('assets/SciFiSounds/laserSmall_003.wav', 'rb') as f:
    beam = f.read()

with open('assets/SciFiSounds/explosionCrunch_001.wav', 'rb') as f:
    explosion = f.read()

with open('assets/SciFiSounds/forceField_003.wav', 'rb') as f:
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
