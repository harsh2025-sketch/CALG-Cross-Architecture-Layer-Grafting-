# CALG: Cross-Architecture Layer Grafting

> **IMPORTANT NOTICE: This project demonstrates structural and systems-level feasibility only. The current architecture is not a production-ready multimodal model and does not yet achieve successful capability transfer.**

CALG (Cross-Architecture Layer Grafting) is an experimental framework for physically integrating heterogeneous foundation-model components into a single executable neural graph without joint pretraining. FusionLM-v0.1-alpha demonstrates the structural feasibility of combining Phi-4, Llama Vision, Qwen-Coder, and Gemma-Math components under a 15 GB VRAM constraint.

## Overview

Cross-Architecture Layer Grafting (CALG) explores whether independently trained foundation models can be physically combined into a single executable neural graph without joint pretraining, instruction tuning, or full-scale backpropagation.

Can structurally incompatible neural networks be mechanically fused and executed as a single runtime architecture using only dimensional projection, low-rank transformation, and localized graph modifications?

To investigate this question, I constructed **FusionLM-v0.1-alpha**, which combines:
* Phi-4-mini-instruct as the host language backbone
* The visual subsystem from Llama-3.2-11B-Vision-Instruct
* Coding-domain task vector deltas derived from Qwen2.5-Coder
* Mathematics-domain task vector deltas derived from Gemma-2-Math

## Website
Read the full technical report and methodology at our [GitHub Pages site](https://username.github.io/calg-fusionlm/).
