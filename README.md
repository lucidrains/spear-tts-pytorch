<img src="./spear-tts.png" width="450px"></img>

## Spear-TTS - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2302.03540">Spear-TTS</a> - multi-speaker text-to-speech attention network, in Pytorch

The text-to-semantic module built here will be used for <a href="https://github.com/lucidrains/soundstorm-pytorch">SoundStorm</a> for conditioning.

## Appreciation

- <a href="https://stability.ai/">Stability</a> and <a href="https://huggingface.co/">🤗 Huggingface</a> for their generous sponsorships to work on and open source cutting edge artificial intelligence research

- <a href="https://github.com/lucasnewman">Lucas Newman</a> for completing the <a href="https://github.com/lucidrains/spear-tts-pytorch/pull/4">backtranslation</a> portion, as well as beam search decoding!

## Todo

- [x] add eos logic + generate, and hook up end-to-end generation in soundstorm
- [x] add first pretraining speech-to-speech with the reconstruction of 60% deleted tokens
- [x] add dropouts for this project, as low-resource

- [ ] add total flexiblity of which layers of encoder / decoder to freeze during training
- [ ] add step for training on small speech -> text corpus and generating pseudo-labelled dataset + finetuning
- [ ] add final step of finetuning on text -> speech + pseudolabelled dataset

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
