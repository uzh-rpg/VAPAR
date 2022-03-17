# GazeSim - Predicting Human Visual Attention in Drone Racing

This repository contains code used for my Master's thesis (which turned into [a paper](https://doi.org/10.1371/journal.pone.0264471)), specifically for training models to predict human visual attention from gaze data of human pilots flying drone racing tracks. It is designed to function with an earlier version of the [Eye Gaze Drone Racing Dataset](https://osf.io/gvdse/), some of which has been replicated with the old structure in the [published dataset for the paper](https://osf.io/uabx4/).

The main use of the code in the context of the paper is however to be able to load trained models for attention prediction. These predictions are then used in for the end-to-end drone racing controller models described in the paper ([code here](https://github.com/swengeler/flightmare)).

Thus, no detailed instructions will be given for the use of the code for its original purpose. The relevant parts are mostly the installation of the Python library for use with the main code repository and the replication of the results from the paper regarding attention prediction. The latter is described in the wiki of the [dataset](). The installation of this Python package simply requires installing the dependencies (using either the `environment.yml` in this repo or that in the [main repo](https://github.com/swengeler/flightmare)), and then running:

```shell
pip install .
```

<!---

The instructions below this line belong to the old `README` and are only included for completeness sake. With the new structure of the [used dataset](https://osf.io/gvdse/), the code will not work.

## Preparing the data for training

Before creating an index and generating ground-truth, individual laps should be filtered by whether they follow an "expected" trajectory (and thus are useful for learning to fly well):
```shell script
python src/data/generate_expected_trajectory_entries.py -tn flat/wave
```
This script will first create an image of all valid trajectories and then overlay each lap individually on top of it, with the user having to choose whether it follows an expected trajectory or not (buttons "Yes" or "No"). For now this has to be executed two times, once for each track name ("flat" and "wave").

To generate a global index of all frames including certain properties to filter by (e.g. `valid_lap`, `expected_trajectory`), run:
```shell script
python src/data/index_data.py
```

After that the "context" for the generation of ground-truth is established, and it an be generated for attention map prediction and control input prediction with the following two commands respectively:
```shell script
python src/data/generate_ground_truth.py -gtt moving_window_frame_mean_gt
```
```shell script
python src/data/generate_ground_truth.py -gtt drone_control_frame_mean_gt
```

To be able to use masked videos, one should first compute the mean mask(s).
--->