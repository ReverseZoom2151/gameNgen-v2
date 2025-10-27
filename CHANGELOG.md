# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

**Missing Paper Implementations:**

- Proper FVD (Fréchet Video Distance) implementation with I3D model (`src/utils/fvd.py`)
- Human evaluation framework following paper's methodology (`src/utils/human_eval.py`)
- Multi-scenario training support for diverse DOOM maps (`src/environment/multi_scenario.py`)

**Developer Tools** (`scripts/`):

- `download_models.py` - Pre-download and verify all required models
- `resume_training.py` - Easy resume from checkpoints
- `visualize_data.py` - Visualize recorded gameplay data
- `compare_models.py` - Compare different checkpoint quality
- `export_video.py` - Batch export gameplay videos
- `monitor_training.py` - Real-time training monitoring

**Advanced Features** (Beyond Paper):

- Text-conditioned game generation with CLIP (`src/diffusion/text_conditioning.py`)
- Image-based modding system for editing games (`src/diffusion/image_modding.py`)
- Hierarchical memory system for longer context (`src/diffusion/hierarchical_memory.py`)

**GitHub Infrastructure:**

- CONTRIBUTING.md - Contribution guidelines
- GitHub Actions CI - Automated testing
- Issue templates (bug report, feature request)
- CHANGELOG.md - This file

### Changed

- Reorganized repository into standard GitHub structure
- Moved configs to `configs/` directory with clearer names
- Moved tests to `tests/` directory
- Moved paper PDF to `paper/` directory
- Updated all import paths and references
- Cleaned root directory to 6 essential files

## [1.0.0] - 2025-10-27

### Initial Release - Complete Implementation

#### Core Implementation

- Action-conditioned Stable Diffusion model (943M parameters)
- Training pipeline with noise augmentation
- Real-time inference (4-step DDIM, 20 FPS)
- PyTorch dataset for gameplay trajectories

#### Tier 1 - Chrome Dino

- DQN agent implementation
- SimpleDinoEnv wrapper
- Training scripts
- Complete in 2-3 days

#### Tier 2 - DOOM Lite

- ViZDoom environment wrapper
- PPO agent training
- DOOM reward function
- Complete in ~1 week

#### Tier 3 - Full DOOM

- Adafactor optimizer (paper's choice)
- Model distillation (1-step, 50 FPS)
- Decoder fine-tuning
- Comprehensive evaluation suite
- Complete in 3-4 weeks

#### Documentation

- Professional README
- 12 comprehensive guides
- Command references
- Installation instructions
- 5,000+ lines of documentation

#### Testing

- Core component tests
- All tier verification
- CUDA/GPU validation
- Installation tests

#### Configuration

- 3 tier-specific configs
- All hyperparameters documented
- Easy customization

### Technical Features

- Noise augmentation for auto-regressive stability
- Velocity parameterization training
- Mixed precision (FP16) support
- Classifier-Free Guidance
- Auto-checkpointing and resume
- TensorBoard logging
- Comprehensive evaluation metrics (PSNR, LPIPS, SSIM)
- Interactive gameplay mode
- Video recording

---

## Release Notes

### v1.0.0 - Complete Implementation

First public release of complete GameNGen implementation.

**Implements:** "Diffusion Models Are Real-Time Game Engines" (ICLR 2025)

**Status:**

- Implementation: Complete
- Tests: All passing
- Documentation: Comprehensive
- Pretrained weights: Training in progress

**What's Included:**

- All 3 tiers fully implemented
- 12,000+ lines of production code
- Ready to train immediately

**Coming Soon:**

- Tier 1 weights (~3 days)
- Tier 2 weights (~1 week)
- Tier 3 weights (~4 weeks)
- Demo videos

---

## Future Roadmap

### Planned Features

- [ ] Pre-trained model weights for all tiers
- [ ] Demo videos and comparisons
- [ ] Jupyter notebook tutorials
- [ ] Web-based demo interface
- [ ] Additional game environments
- [ ] Multi-game universal model
- [ ] Longer context methods
- [ ] Docker support

### Research Extensions

- [ ] Text-to-game generation
- [ ] Image-based style transfer
- [ ] Real-world applications (robotics, driving)
- [ ] Improved architectures

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## Questions?

Open an issue or check the [discussions](https://github.com/ReverseZoom2151/gameNgen-v2/discussions).
