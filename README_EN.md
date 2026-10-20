<div align="center">

# SpektraDarkroom Vulkan Edition

### Professional 35mm Physical Film Development and Optical Print Darkroom Workstation (Windows)

[**中文文档**](README.md) | [**English Documentation**](README_EN.md)

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Profiles: CC BY-SA 4.0](https://img.shields.io/badge/Profiles-CC%20BY--SA%204.0-orange.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Platform: Windows 10 / 11](https://img.shields.io/badge/Platform-Windows%2010%20%2F%2011%20(64--bit)-0078D6.svg?logo=windows)](https://www.microsoft.com)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python)](https://www.python.org)
[![Hardware Acceleration: Modern OpenGL / Vulkan](https://img.shields.io/badge/Hardware%20Acceleration-OpenGL%20%2F%20Vulkan-ED1B24.svg?logo=vulkan)](https://www.vulkan.org)

<br/>

![SpektraDarkroom Main Workbench Preview](img/main_ui_preview.png)

</div>

---

## Overview

**SpektraDarkroom** is a desktop workstation for 35mm physical film development and optical print simulation on Windows. This project was originally conceived and built for personal darkroom development and film grading workflows, with the goal of creating a standalone, high-framerate, non-destructive desktop tool equipped with a complete physical darkroom pipeline and intuitive controls.

Built upon continuous spectral simulation and photochemical sensitometry/development models, SpektraDarkroom faithfully recreates every stage from camera RAW negative exposure and DIR inhibitor coupler inter/intra-layer diffusion, to subtractive dichroic enlarger filter exposure, photographic paper development, and authentic silver halide dye-cloud grain physics.

---

## Key Features

### 1. Physical Darkroom Continuous Spectral Simulation
- **81-Band Continuous Spectral Sensitometry**: Computes physical film exposure and absorption across 81 spectral bands, eliminating color shifts and banding artifacts common to traditional 3D LUTs.
- **52 Film Stock Presets + 9 Print Paper & Projection Stock Profiles**:
  - **Classic Color Negatives**: Kodak Portra 160 / 400 / 800, Gold 200, ColorPlus 200, UltraMax 400, Ektar 100; Fujifilm Pro 400H / 160NS, C200, Superia X-TRA 400, etc.
  - **Motion Picture Stocks**: Kodak Vision3 50D / 200T / 250D / 500T, CineStill 50D / 400D / 800T, Fujifilm Eterna 500T / Vivid 500T, F-64D, etc.
  - **Legendary Color Reversals (Slides)**: Fujifilm Velvia 50 / 100, Provia 100F, Astia 100F, Kodak Ektachrome E100, Kodachrome 25 / 64 / 200, Agfachrome CT 50 / 100 / 200, etc.
  - **Classic B&W Negatives**: Kodak Tri-X 400, T-Max 100, Ilford HP5 Plus 400, FP4 Plus 125, Delta 3200, Fujifilm Neopan Acros 100, Kodak Double-X 5222, etc.
  - **Photographic Papers & Print Stocks**: Kodak Vision 2383 / 2393 motion picture print films, Kodak Endura Premier, Portra Endura, Ultra Endura, Supra Endura, Ektacolor Edge, Fujifilm Crystal Archive II, and direct digital scan inversion modes.
- **Subtractive Dichroic Enlarger Filter Pack**: Accurate simulation of stepless Cyan, Magenta, and Yellow dichroic filters, featuring an integrated Neutral Print Filters Table for automatic neutral density compensation.
- **Authentic Film Grain & Dye Clouds (CLT Model)**: Viewport-integrated decay model based on the Central Limit Theorem (CLT), simulating true 35mm C-41 dye clouds with decoupled per-emulsion RGB layer sampling. Delivers smooth integration at macro zoom and crisp micro-grain definition at 1:1 view.
- **Photochemical Optics**: Non-linear red halation reflection diffusion, DIR development inhibitor coupler effects, darkroom pre-flashing, and film format grain scaling.

### 2. Low-Latency Viewport & GPU Hardware Acceleration
- **Modern OpenGL / Vulkan Acceleration**: Real-time shader-level 3D physical color space mapping, driving 4K viewports at 60+ FPS responsive interactions.
- **Smooth Zoom & Anchor Tracking**: Smooth zoom range from 10% to 800%, with cursor anchor targeting on double-click magnification, and quick 1:1 / fit-to-view toggles.
- **Inspection & Comparison Tools**: Split-screen draggable viewport slider, and instant raw comparison via hold-key (`\`).
- **Multi-Tier Export Engine**: Supports both high-precision off-screen CPU export and GPU framebuffer hardware-accelerated batch export.

### 3. Professional Darkroom Batch Productivity
- **Native RAW Format Support**: Supports ARW, CR2, CR3, NEF, RAF, DNG, and 32-bit float / 16-bit lossless TIFF and PNG.
- **Non-Destructive Sidecar Architecture (.sdc)**: Grading and darkroom parameters are automatically persisted into companion `.sdc` files, ensuring original source negatives remain untouched.
- **Session Workspace Management (.sdss)**: Save and restore entire multi-frame darkroom sessions for reproducible workflows and cross-machine handoffs.
- **Multi-Frame Batch Operations**: Ctrl+Click / Shift+Click range selections, full parameter copy-paste across frames, and asynchronous multi-threaded pipeline export.
- **Memory Protection & Dual-Tier Caching**: Tier 1 thumbnail cache and Tier 2 viewport texture cache with strict LRU memory reclamation, guaranteeing stable memory consumption during continuous heavy RAW browsing.
- **Precision Input Controls**: Drag value labels for fine step adjustments, dedicated keyboard steppers, and full Undo/Redo (Ctrl+Z / Ctrl+Y) history stacks.

---

## System Requirements

* **Operating System**: Windows 10 / Windows 11 (64-bit)
* **Python Version**: Python 3.10 - 3.13 (64-bit)
* **Graphics Hardware**: Dedicated or integrated GPU supporting OpenGL 3.3+ and Vulkan compatible display drivers (NVIDIA / AMD / Intel)

---

## Getting Started & Running from Source

### 1. Clone the Repository
```bash
git clone https://github.com/wwppp/spektra-darkroom.git
cd spektra-darkroom
```

### 2. Set Up Python Virtual Environment (Recommended)
```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Or Windows Command Prompt (cmd)
.\.venv\Scripts\activate.bat
```

### 3. Install Dependencies
All core simulation modules are built-in. Install the required packages via:
```bash
pip install -r requirements.txt
```

### 4. Launch Application
- **Direct Launcher** (Recommended, no console window): Double-click **`SpektraDarkroom.exe`** in the root directory.
- **Or via Command Line**:
  ```bash
  python main.py
  ```

---

## Licenses

This project adheres to the following licensing terms:

1. **Source Code**: Released under the **[GNU General Public License v3.0 (GPLv3)](LICENSE)**.
2. **Physical Film & Paper Profiles (`resources/profiles/`)**: Digitized and structured from continuous spectral sensitometry datasheets, licensed under **[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)**.
3. **Derived 3D LUT Assets**: Free to use and share; individual standalone resale or commercial repackaging is prohibited.

---

## Credits & Acknowledgments

- **[Andrea Volpato (spektrafilm)](https://github.com/andreavolpato/spektrafilm)**: Sincere gratitude to Andrea Volpato for pioneering `spektrafilm`. His rigorous color science research on continuous spectral emulsion models, coupler chemical development reactions, and neutral gray print matrices forms the foundation of this software.
- **[fotufilm-engine](https://github.com/fotufilm/fotufilm-engine)**: Sincere gratitude to `fotufilm-engine`. We adopted and refined its multi-layered dye-cloud grain physics, emulsion micro-structure parameters, and high-precision classic film profile datasets, elevating grain realism and profile coverage.
- **[James Qiu (SpektraLab)](https://github.com/JamesQiu2005/SpektraLab)**: Acknowledgement for inspiring engineering approaches in darkroom modernization.
- **[pixls.us Free Software Photography Community](https://discuss.pixls.us)**: Many thanks to the open photography community for open discussions and shared datasheets in physical darkroom simulation.
