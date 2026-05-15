# Seeing Good Keypoints Through Water

This repository contains the code for my TTIC 31040 final project on robust local-feature detection under dynamic water-surface distortion.

The project studies whether SIFT-style keypoints can be made more stable when images are distorted by a moving refractive water surface. I implemented a physics-inspired water distortion simulator, an sRD-SIFT-inspired radial distortion model, and an optical-flow-based keypoint pruning method.

## Summary

Under high-distortion settings, the combined distortion-aware SIFT + optical-flow filtering pipeline improved keypoint match repeatability and recall by approximately 10% over baseline SIFT. Precision did not consistently improve.

## Method

The pipeline has four main components:

1. **Dynamic water-surface simulator**  
   Simulates refractive image distortion using random traveling sine waves and Snell-style ray bending.

2. **Radial distortion approximation**  
   Uses an sRD-SIFT-inspired radial model to approximate refraction-induced distortion.

3. **Forward-backward optical-flow pruning**  
   Tracks candidate keypoints across simulated frames and removes unstable points with high forward-backward error.

4. **Evaluation framework**  
   Measures repeatability, recall, and precision under multiple distortion amplitudes and camera/water-depth settings.

## Results

The main experiment ran 600 trials across 10 distortion settings.

| Method | Repeatability | Recall | Precision |
|---|---:|---:|---:|
| Baseline SIFT | baseline | baseline | baseline |
| Proposed method | +~10% | +~10% | no consistent gain |

The improvement was strongest in the high-distortion regime.

<img width="529" height="554" alt="image" src="https://github.com/user-attachments/assets/cac8d1b7-8cbd-49d6-9a79-781120f2e651" />

## Method Demo

First frame contains all keypoints detected using rdSIFT. Following frames are only the keypoints that survived forwards-backwards optical-flow pruning.

https://github.com/user-attachments/assets/16237e54-096f-41c3-83c2-fe6e547a2500

## Repository Structure

```text
project/
  water_surface_simulator.py   # dynamic refractive water simulator
  radial.py                    # radial distortion model
  srd_sift.py                  # modified SIFT implementation
  keypoint_flow.py             # forward-backward optical-flow pruning
  matching.py                  # repeatability / recall / precision metrics
  hpatches_images.py           # loading testing set

scripts/
  run_flow_eval.py             # main dynamic-water evaluation
  run_static_eval.py           # static radial-distortion evaluation
  plot_results.py              # aggregate and plot experiment results
  approx_diff.py               # estimate error between true distortion and radial model
'''


