# Narration script

Timings measured from `ICRA_2027.mp4` (audio-aligned, faster-whisper `medium.en` + forced alignment to this text).

| # | start | end | shot | narration |
|---|-------|-----|------|-----------|
| s01 | 0:00.00 | 0:04.92 | s01_title | In this work, we present how to guarantee human safety in human-robot collaboration from camera images. |
| s02 | 0:08.10 | 0:25.82 | s02_dangerous_failure | If our prediction is wrong, we speak of a failure. This failure is dangerous if the robot would actually touch the human in this instance. The ISO standard requires safety systems to have a probably of dangerous failures per hour of less than one in a million, which is at most one dangerous failure every 114 years. To achieve this, we propose the following system. |
| s03 | 0:25.82 | 0:30.78 | s03_pose2d | First, our 2D pose estimation predicts the joint positions and covariance matrices. |
| s04 | 0:31.18 | 0:35.72 | s04_triangulate | Our uncertainty-aware triangulation lifts the poses and covariances into three dimensions. |
| s05 | 0:39.78 | 0:48.42 | s05_motion | We then input two seconds of pose history in our motion prediction transformer to predict the future poses and their covariances in the next 400 milliseconds. |
| s06 | 0:50.78 | 1:03.10 | s06_conformal | To establish strong probabilistic guarantees, we use conformal calibration: on a calibration dataset, we define the score as the distance between the prediction and ground truth in units of predicted sigma. We set the score threshold to the 99.99th percentile that scales every covariance matrix into a conformal prediction set. **[not in the audio: “on a calibration dataset, we define the score as the distance between the prediction and ground truth in units of predicted sigma.”]** |
| s08 | 1:15.90 | 1:28.54 | s08_shield | The spheres now become capsules, which define the human's reachable occupancy. SARA shield intersects them with the robot's reachable occupancy. If there is no intersection, we execute the trajectory. Otherwise, a failsafe stop. |
| s07 | 1:30.68 | 1:37.94 | s07_ood | To handle rare OOD inputs, we detect them with gradient-based monitors and reuse the previous prediction to resume operation. |
| s09 | 1:42.32 | 1:55.70 | s09_vs_iso | Our experimental results on the human 3.6 million dataset show that our reachable occupancies in blue have a seven point six times smaller volume than the constant-velocity model of ISO thirteen eight fifty-five in orange. |
| s11 | 1:56.04 | 2:06.36 | s11_sim_setup | To determine the probability of dangerous failures per hour of our system, we take every prediction and randomly place it within a ten-meter circle around the robot 265 billion times. |
| s12 | 2:09.18 | 2:15.54 | s12_sim_prune | Almost none of those placements can reach the robot; hence we pre-filter them and evaluate the remaining motions on the GPU. |
| s14 | 2:17.38 | 2:35.46 | s14_pfhd | For our system, four of those placements ended in contact although the shield had verified the trajectory. Using a Clopper-Pearson bound, we can say with 99.999 percent confidence that our system has a dangerous failure rate below the required 1 in a million failures per hour. |
| s16 | 2:35.76 | 2:52.80 | s16_realworld | In our hardware experiments we evaluate our pipeline on a Franka arm and a RealSense camera at twenty-five hertz. Blue is the human's conformal prediction sets over the robot's stopping horizon. When they intersect the robot's reachable set, the shield brakes. In every tested instance the robot came to a complete stop before the human could reach it. |
| s17 | — | — | s17_outro |  |
