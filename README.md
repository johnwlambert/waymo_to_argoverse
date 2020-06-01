# waymo_to_argoverse
Waymo Open Dataset -> Argoverse

Simple utility to convert Waymo Open Dataset raw data, ground truth, and detections to the Argoverse format, run an Argoverse-based tracker, and then submit to leaderboard.

Achieves the following on the Waymo 3d Tracking Leaderboard, using `run_ab3dmot.py` from my [argoverse_cbgs_kf_tracker](https://github.com/johnwlambert/argoverse_cbgs_kf_tracker) repo.

|    Model                | MOTA/L2    | 	MOTP/L2   | 	FP/L2	  |   Mismatch/L2	|   Miss/L2  |
| :---------------------: | :-------:  | :--------: | :--------:| :--------:    | :--------: |
| PPBA AB3DMOT (mine)     | 0.2914	   |  0.2696	  | 0.1714    |	0.0025 	      | 0.5347     |
| Waymo Baseline          |  0.2592	   | 0.1753	    | 0.0932    |	0.0020	      |  0.3122    |



	      
PerType	


Submission process overview is [here](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md#use-pre-compiled-pippip3-packages).
