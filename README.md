# PicoDet — quick CPU check (Sept 2025)

**Goal**  
Sanity-check PicoDet (COCO weights) for spotting **drones** in video, with Raspberry-Pi–class CPU in mind.

**Setup**  
- Machine: Mac M3 (CPU, `env_id=0`)  
- Models tried: `s-416`, `m-416`, `l-640`  
- Inputs: `solakair/clips/` → Outputs: `solakair/results/`  
- How to run: see `cmds.txt` (exact commands) and `env.txt` (package versions)

> **Note:** COCO has **no “drone” class**. Predictions come as *bird / airplane / kite*; a small heuristic sometimes relabels as **“drone?”**.

## What I saw
- **640** input improves tiny-object recall a bit (slower FPS).
- Mislabels are common for small/far drones; confidence flickers frame-to-frame.
- Lighting/background matter a lot. A simple tracker would stabilize boxes.

## Full results (videos)
All output clips (full-length) are here:  
**Google Drive:** https://drive.google.com/drive/folders/1sAkaQjZHFu9vgKbcyp7bDEdWzgcVIrvX?usp=sharing

> Tip: if the link ever changes, update `manifest.csv` to keep filenames in sync.


## Takeaway
Good **baseline on CPU**, but for drone use we should **fine-tune a small detector on drone data** and export **ONNX**.

## Next steps
1. Fine-tune (PicoDet or YOLO-Nano) on a drone-only dataset (e.g., VisDrone + negatives).  
2. Export ONNX and plug into this runner; measure **CPU FPS & accuracy**.  
3. Add multi-frame smoothing / ROI or a tracker (ByteTrack / DeepSORT).  
4. Try on **Raspberry Pi 5** with real camera footage.

*Models auto-download on first run (internet required).*  
*Ailia SDK license: https://ailia.ai/license/*
