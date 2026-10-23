# Neural cellular automata — self-healing lizard

![The lizard growing from one pixel, getting cut, and growing back](figures/heal.gif)

## What it is

A tiny neural network that lives in every pixel of a 72x72 grid. Each pixel can only see the pixels right next to it, but starting from one single pixel they grow a lizard emoji together, and if you cut part of it off they grow it back. The whole network is only **8,320 weights**. You can [try it in your browser](https://menzer-david.github.io/neural-cellular-automata/demo/) and cut it yourself.

I trained three versions of the lizard and tested how well each one heals. The one that got cut up during training healed **70%** of medium-sized holes. The one that never got cut healed 25%, and the one that only learned to grow healed none. There's also a butterfly and a mushroom in the demo.

## Why I built it

I recently started learning how neural networks train by building backprop from scratch with Karpathy's micrograd. I wanted a project where I could actually see a neural network working instead of just getting an accuracy number. When I read about Growing Neural Cellular Automata, a paper where one tiny network copied into every pixel grows a shape and repairs it like cells in a body, I wanted to build it myself.

## How to run it

Needs Python 3 with `torch`, `numpy`, `pillow`, and `matplotlib`. The trained models are already in `checkpoints/`, so you can just open `demo/index.html` in a browser. To train your own, run `python3 nca.py train --target lizard --iters 3000 --out checkpoints/lizard.pt`. Training takes about an hour per model on my M1 MacBook. `python3 experiments.py` remakes all the graphs in `figures/`.

## What I learned

This project taught me that the way you train a model matters just as much as the model itself, because all three lizards had the same network but only the one that practiced getting cut could really heal. I also learned how much memory backprop uses when you run it through 80 steps, and I had to use gradient checkpointing to fit it on my laptop. And I learned to keep my charger plugged in, because my battery died halfway through training.

## Credits

Based on [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) by Mordvintsev et al. (Distill, 2020). Emoji images from Google's [Noto Emoji](https://github.com/googlefonts/noto-emoji) (license in `targets/LICENSE`).
