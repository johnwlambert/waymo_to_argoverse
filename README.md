
## Waymo Open Dataset -> Argoverse Converter

### Overview

Simple utility to convert Waymo Open Dataset raw data, ground truth, and detections to the Argoverse format [ [paper](https://arxiv.org/abs/1911.02620), [repo](https://github.com/argoai/argoverse-api) ], run a tracker that accepts Argoverse-format data, and then submit to Waymo Open Dataset leaderboard.

Achieves the following on the Waymo 3d Tracking Leaderboard, using `run_ab3dmot.py` from my [argoverse_cbgs_kf_tracker](https://github.com/johnwlambert/argoverse_cbgs_kf_tracker) repo.

|    Model                    | MOTA/L2    | 	MOTP/L2   | 	FP/L2	  |   Mismatch/L2	|   Miss/L2  |
| :-------------------------: | :-------:  | :--------: | :--------:| :--------:    | :--------: |
| HorizonMOT3D                | 0.6345     | 0.2396     | 0.0728    | 0.0029        | 0.2899     |
| PV-RCNN-KF                  | 0.5553     | 0.2497     | 0.0866    | 0.0063        | 0.3518     |
| Probabilistic 3DMOT         | 0.4765     | 0.2482     | 0.0899    | 0.0101        | 0.4235     |
|            ...              |   ...      |    ...     |    ...    |     ...       |   ...      |
| **PPBA AB3DMOT (this repo)**| **0.2914** |  0.2696	  | 0.1714    |	0.0025 	      | 0.5347     |
| Waymo Baseline              |  0.2592	   | 0.1753	    | 0.0932    |	0.0020	      |  0.3122    |


### Usage Instructions for Waymo Leaderboard

1. Download test split files (~150 logs) from [Waymo Open Dataset website](https://waymo.com/open/download/) which include TFRecords.
2. Download provided detections from PointPillars Progressive Population-Based Augmentation detector, as .bin files.
3. Convert to Argoverse format using scripts provided here in this repo.
4. Run tracker
5. Convert track results to .bin file
6. Populate a `submission.txtpb` file with metadata describing your submission ([example here](https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/waymo_open_dataset/metrics/tools/submission.txtpb)).
7. Run `create_submission` binary to get tar.gz file. Binary is only compiled using Bazel. I used Google Colab. 
8. Submit to [Waymo eval server](https://waymo.com/open/challenges/3d-tracking/).


Submission process overview is [here](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md#use-pre-compiled-pippip3-packages).
