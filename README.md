
## Waymo Open Dataset -> Argoverse Converter

### Repo Overview

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


## Data Format Overview

Waymo raw data follows a rough class structure, as defined in [Frame protobuffer](https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto).
Waymo labels and the detections they provide also follow a rough class structure, defined in [Label protobuffer](https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto).

Argoverse also uses a notion of Frame at 10 Hz, but only for LiDAR and annotated cuboids in LiDAR. This is because Argoverse imagery is at 30 Hz (ring camera) and 5 Hz (stereo). Argoverse data is provided at integer nanosecond frequency throughout, whereas Waymo mixes seconds and microseconds in different places. **Argoverse LiDAR points are provided directly in the egovehicle frame, not in the LiDAR sensor frame, as [.PLY](http://paulbourke.net/dataformats/ply/) files.**

A Waymo object defines a coordinate transformation from the labeled object coordinate frame, to the egovehicle coordinate frame, as an SE(3) comprised of rotation (derived from heading) and a translation:
```python
object {
  box {
    center_x: 67.52523040771484
    center_y: -1.3868849277496338
    center_z: 0.8951533436775208
    width: 0.8146794438362122
    length: 1.8189797401428223
    height: 1.790642261505127
    heading: -0.11388802528381348
  }
  type: TYPE_CYCLIST
}
score: 0.19764792919158936
context_name: "10203656353524179475_7625_000_7645_000"
frame_timestamp_micros: 1522688014970187
```

Argoverse data is provided similarly, but in JSON with full 6 dof instead of 4 dof transformation from labeled object coordinate frame to egovehicle frame. A quaternion is used for the SO(3) parameterization:
```python
{
  "center": {"x": -25.627050258944625, "y": -3.6203567237860375, "z": 0.4981851744013227}, 
  "rotation": 
    {"x": -0.000662416717311173, 
    "y": -0.000193607239199898, 
    "z": 0.000307246307353097, "w": 0.999999714659978}, 
    "length": 4.784992980957031, 
    "width": 2.107541785708549, 
    "height": 1.8, 
    "track_label_uuid": "215056a9-9325-4a25-bbbd-92d445d60168", 
    "timestamp": 315969629119937000, 
    "label_class": "VEHICLE"
},
```
Whereas Waymo uses "context.name" as a unique log identifier, Argoverse uses "log_id".

### Guide to Repo Code Structure
- `waymo_dets_to_argoverse.py`: Convert provided Waymo detections to Argoverse format. Use shards to not exceed Colab RAM.
- `dump_waymo_persweep_detections.py`: Given sharded JSON files containing labeled objects or detections in random order, accumulate objects according to frame, at each nanosecond timestamp. Write to disk.
- `waymo_data_splits.py`: functions to provide list of log_ids's in Waymo val and test splits, respectively.
- `waymo_raw_data_to_argoverse.py`: Extract poses, images, and camera calibration from raw Waymo Open Dataset TFRecords.
- `run_tracker.sh`: script to run [AB3DMOT-style tracker](https://github.com/johnwlambert/argoverse_cbgs_kf_tracker) on Argoverse-format detections, and write tracks to disk. 
- `create_submission_bin_file.py`: Given tracks in Argoverse format, convert them to Waymo submission format.

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


## References
```
@InProceedings{Chang_2019_CVPR,
author = {Chang, Ming-Fang and Lambert, John and Sangkloy, Patsorn and Singh, Jagjeet and Bak, Slawomir and Hartnett, Andrew and Wang, De and Carr, Peter and Lucey, Simon and Ramanan, Deva and Hays, James},
title = {Argoverse: 3D Tracking and Forecasting With Rich Maps},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
