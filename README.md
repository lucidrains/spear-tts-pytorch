## Spear-TTS - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2302.03540">Spear-TTS</a> - multi-speaker text-to-speech attention network, in Pytorch

The text-to-semantic module built here will be used for <a href="https://github.com/lucidrains/soundstorm-pytorch">SoundStorm</a> for conditioning.

## Todo

- [ ] add eos logic + generate, and hook up end-to-end generation in soundstorm
- [ ] add total flexiblity of which layers of encoder / decoder to freeze during training
- [ ] add first pretraining speech-to-speech with the reconstruction of 60% deleted tokens

## Citations

```bibtex
@misc{kharitonov2023speak,
    title   = {Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision}, 
    author  = {Eugene Kharitonov and Damien Vincent and Zalán Borsos and Raphaël Marinier and Sertan Girgin and Olivier Pietquin and Matt Sharifi and Marco Tagliasacchi and Neil Zeghidour},
    year    = {2023},
    eprint  = {2302.03540},
    archivePrefix = {arXiv},
    primaryClass = {cs.SD}
}
```
