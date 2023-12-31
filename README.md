# Galactic Harvesters: The Canister Crusade

This is [my entry](https://pyweek.org/e/szabolcsdombi) for [PyWeek 36](https://pyweek.org/36/).

```
pip install -r requirements.txt
```

Download the [assets.pickle](https://github.com/szabolcsdombi/pyweek-36/releases/download/2023-09-23/assets.pickle) and place it next to the `run_game.py`

```
python run_game.py
```

- [download](https://github.com/szabolcsdombi/pyweek-36/releases)
- [watch](https://youtu.be/04alBvihDqk)

## Troubleshooting

Ubuntu users need a [very sad audio fix](https://github.com/szabolcsdombi/pyweek-36/commit/5470a616faa9f2b2cd9f3a9b115efac1225fd303) for https://github.com/szabolcsdombi/pyweek-36/issues/1 and https://github.com/szabolcsdombi/pyweek-36/issues/2

For the fix please install [OpenAL](https://www.openal.org/) and [modernal](https://pypi.org/project/modernal/)

```
sudo apt-get install libopenal-dev
pip install modernal==0.9.0
```

It is possible to play without audio:

```
python run_game.py --no-audio
```

It is possible to play without fullscreen:

```
python run_game.py --no-fullscreen
```

It is possible to play without a mouse:

Move: <kbd>W</kbd> <kbd>A</kbd> <kbd>S</kbd> <kbd>D</kbd>
Turn: <kbd>Q</kbd> <kbd>E</kbd>
Shoot: <kbd>CTRL</kbd>

It is possible to reset the game by deleting the score.txt

It is possible to unlock all ships by modifying the score.txt

It is not possible to play without the asset file.

## Story

In the year 3077,
the Milky Way Galaxy is in the midst of an energy crisis.

The primary source of energy, a rare crystalline element called **"Dark Matter"** is nearing depletion.
Dark Matter is primarily stored in canisters that have been scattered throughout space over centuries
due to space wars, trading routes, and exploration mishaps.

Captain Neil Starbreaker is the fearless pilot of the spacecraft "Nebula Harvester".
Neil used to be a space pirate but has since reformed after witnessing
the dire effects of the energy crisis on his home planet, Noverra.

Join Captain Starbreaker on the "Nebula Harvester" and help save the galaxy.

## Modules

- [pyglet](https://github.com/pyglet/pyglet)
- [zengl](https://github.com/szabolcsdombi/zengl)
- [pyglm](https://github.com/Zuzu-Typ/PyGLM)

## Music

- [Impact Prelude](https://filmmusic.io/song/7565-impact-prelude)
- [Beauty Flow](https://filmmusic.io/song/5025-beauty-flow)

## Assets

- [Space Kit](https://www.kenney.nl/assets/space-kit)
- [Planets](https://www.kenney.nl/assets/planets)
- [Sci-Fi Sounds](https://www.kenney.nl/assets/sci-fi-sounds)
- [Simple Space](https://www.kenney.nl/assets/simple-space)

![assets-1](https://github.com/szabolcsdombi/pyweek-36/assets/11232402/1e956c8a-1f73-40bf-b875-f62d0b4bfd62)

![assets-2](https://github.com/szabolcsdombi/pyweek-36/assets/11232402/5fbd3475-3609-443b-b970-af8a40ba7fac)

![assets-3](https://github.com/szabolcsdombi/pyweek-36/assets/11232402/9c44fc04-625b-4ead-b429-60eb699c868b)

## Devlog

### Day 1

![day-1](https://github.com/szabolcsdombi/pyweek-36/assets/11232402/cc73e02c-61a4-4b37-8894-9f576f6e66d7)

### Day 2

![day-2](https://github.com/szabolcsdombi/pyweek-36/assets/11232402/25510a89-b71f-418e-b8a2-e645a1b5fdbc)

## Day 3

![day-3](https://github.com/szabolcsdombi/pyweek-36/assets/11232402/2e12915d-b7ca-4d9f-a498-630e6db947d6)

## Day 4

![day-4a](https://github.com/szabolcsdombi/pyweek-36/assets/11232402/c349c921-2af6-4dbe-8d5e-2ce0744b90f5)
![day-4b](https://github.com/szabolcsdombi/pyweek-36/assets/11232402/38ae5db0-fc54-4d8a-876e-e080cf03e33f)
![day-4c](https://github.com/szabolcsdombi/pyweek-36/assets/11232402/55d675ac-be0e-4bc8-8149-31400771d6ae)

## Day 5

https://github.com/szabolcsdombi/pyweek-36/assets/11232402/107a5568-6636-49ed-8dd7-61a0c70e3769
