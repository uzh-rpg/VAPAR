# VAPAR

**VAPAR** is a flight trajectory and RGB camera dataset in autonomous drone racing scenarios available on [OSF](https://osf.io/uabx4/).

If you use the code or data in an academic context, please cite the following work:

APA:
```
Pfeiffer C, Wengeler S, Loquercio A, Scaramuzza D (2022) Visual attention prediction improves performance of autonomous drone racing agents. PLOS ONE 17(3): e0264471. https://doi.org/10.1371/journal.pone.0264471
```

Bibtex:
```
@article{10.1371/journal.pone.0264471,
    doi = {10.1371/journal.pone.0264471},
    author = {Pfeiffer, Christian AND Wengeler, Simon AND Loquercio, Antonio AND Scaramuzza, Davide},
    journal = {PLOS ONE},
    title = {Visual attention prediction improves performance of autonomous drone racing agents},
    year = {2022},
    month = {03},
    volume = {17},
    url = {https://doi.org/10.1371/journal.pone.0264471},
    pages = {1-16},
    number = {3},
}
```

# Structure

This repository is structured to mirror the main contributions of the paper mentioned above: 

1. Predicting visual attention from human eye gaze data during drone racing
2. Training an end-to-end controller for flying a race track in simulation using attention features (and comparing the performance to two baseline models)

## Attention prediction

The `attention-prediction` folder contains data to replicate the results of the trained attention model(s) presented in the paper. 

While the data stems from the [Eye Gaze Drone Racing Dataset](https://osf.io/gvdse/), the code was written for an earlier version of this dataset.

### Data

Thus a minimal version of the dataset in the old structure (with some preprocessed data as well) is provided here. All of this data is included in the `data` subfolder. 

This includes the following files for each experiment run (under the path `s<subject-id>/<run-id>_<track-name>`), with each being marked as included in the [published dataset](https://osf.io/gvdse/) with new structure (**O**) or generated using code linked below (**G**):

 - `drone.csv` (**O**): The drone trajectory data needed to potentially create video data using Flightmare (more on this later)
 - `screen_timestamps.csv` (**O**): An index to synchronise video frames with other timestamped data (e.g. trajectory data)
 - `moving_window_frame_mean_gt.mp4` (**G**): Attention map ground-truth data in the form of videos where each frame matches with the original FPV video
 - `shuffled_random_moving_window_frame_mean_gt.mp4` (**G**): Same as the above, but for each lap around the race track, the "frames"/attention maps are shuffled (this is used as a baseline to compare model performance to)
 - `flightmare_60.mp4` (**G**): First-person videos rendered in the Flightmare simulator using the drone trajectories (only provided for test data for easy replication of the results)

In addition, the `index` folder contains `frame_index.csv` (**G**), which is a global index over all frames in the dataset and their properties. For our purposes this index is mostly used to separate the data into a training and test split, but it contains properties for each frame as well (e.g. which lap of an experiment run they are from).

This train/test split is provided in the `splits` folder under `split011.csv` (**G**), in which each row matches a frame in `frame_index.csv` and labeled as training set (`"train"`) or test/validation set (`"val"`).

Matching this particular split of the data is `split011_mean_mask.png` (**G**) in the `preprocessing_data` folder, which is the mean attention map over all generated attention maps (`moving_window_frame_mean_gt.mp4`) in the training set. It is used as a baseline to compare model performance to.

### Model checkpoints

The `checkpoints` folder contains the trained model parameters (and configuration files) for the two attention prediction models compared in the paper (`resnet` and `deep-supervision`). The structure of the subfolders is such that the models can be loaded easily with the provided code.

## End-to-end controllers

The `controller` folder contains data to replicate the results of the trained end-to-end controller models presented in the paper.

Since we compare three different models - with different (abstractions of) visual inputs - the subfolders for this part of the dataset are generally divided into data for the image-based (`images`), feature-track-based (`feature-tracks`) and (attention) encoder-features-based (`encoder-features`) models.

### Data

The `data-training` folder contains all of the data generated during the process of training models using imitation learning. As described in the paper, training trajectories are flown using actions from an MPC expert (and/or the model being trained) over multiple rollouts. This data is recorded and then used to train a model, which gradually takes over more control in future rollouts. This training data can in principle be used to train models fully offline, although this will (very likely) give different results than the fully trained models we provide. Also provided are `config.yaml` files, which contain all of the model and training parameters.

Data resulting from testing the final trained models on both training and (unseen) test trajectories can be found in the `data-testing` folder. For each model, results of repeatedly flying each of these 36 trajectories are provided in their own subfolder. 

These folders are labeled `trajectory-flat-s<subject-id>-<split>_r<run-id>_li<lap-index>_buffer20`, identifying the original subject, experiment run, and lap index from the attention dataset described above. The `buffer20` portion simply states that the trajectory is "extended" by 2 seconds at the start, over which the MPC will always initiate the flight.

For each of these trajectory folders, the recorded trajectories of 10 flights following the reference trajectory (included under `original.csv`) are provided. These are labeled `mpc2nw_mt-<max-time>_st-<switch-time>_<repetition-id>.csv`, where the maximum time depends on the length (in time) of the trajectory, the switch time is always 2 seconds, and the repetition ID signifies independent repetitions. In addition, `mpc_nw_act_00.csv` is included, which contains data from the entire trajectory being flown by the MPC expert, with the actions predicted by the network also being recorded. For the `feature-tracks` and `encoder-features` models a `feature-data.zip` file is also included, providing the feature track/(attention) encoder feature data used as input for model predictions for each recorded data point in the CSV-files.

### Inputs

The `inputs` folder mainly contains the training and test trajectories used for training and evaluating controllers (under `mtt_total_median/flat`) and the trained attention model (under `attention_model_0`), which is used to extract (attention) encoder features as input to the `encoder-features` model. An addition gaze prediction model is also included under `attention_model_1`, although it is not used for any results in the paper.

### Model checkpoints

The final parameters of the three models are included in the `checkpoints` folder and can e.g. be used to replicate our test results. The important files here are mostly in the `model` folder (under which the model weights and structure are saved) and the `config.yaml` file.

# Installation

The code for the results in the paper is similarly split into two parts. The [GazeSim repository](https://github.com/swengeler/gazesim) provides the code for the attention prediction model(s). Since parts of it are also needed for loading the attention model for the end-to-end controllers, the Python package has to be installed before being used with the code for the latter in a [customized Flightmare repository](https://github.com/swengeler/gazesim).

Installation instructions are provided in the `README` files of each of the repositories (particularly in the end-to-end controller repo). However, for more clarity, these are the main steps for getting the entire code base to run:

1. Install all Python dependencies (it is easiest to use the `environment.yml` file [here](https://github.com/swengeler/flightmare/blob/master/environment.yml))
2. Follow the [official instructions](https://flightmare.readthedocs.io/en/latest/) for non-Python dependencies for installing the Flightmare simulator
3. Download the Flightmare Unity binaries from [here](https://github.com/swengeler/flightmare_unity) (or use that repository and Unity to build them yourself)
4. Install the `gazesim` Python package using [its repository](https://github.com/swengeler/gazesim)
5. Follow the installation instructions for the [customized version of Flightmare](https://github.com/swengeler/gazesim) used for this project
 
## Replicating results

While we do not provide instructions for using the provided code in detail, as a starting point, here are some general instructions for replicating our results.

### Attention prediction

To simply confirm the test scores of our attention model(s) and baselines as described in the paper, the full test data is included in the data set (see above). Once the dataset has been downloaded and the environment variable `GAZESIM_ROOT` has been set to the location of the `attention-prediction/data` folder, the `eval_attention.py` script (in `gazesim/analysis` in the repository) can be used to evaluate the models. It could e.g. be called as such (for more options, see the code):

```shell
python eval_attention.py -of <output-file> -mm attention-prediction/data/preprocessing_info/split011_mean_mask.png -m attention-prediction/checkpoints/resnet/model/resnet.pt attention-prediction/checkpoints/deep-supervision/model/deep-supervision.pt
```

This will evaluate the two provided model checkpoints, as well as the mean attention map and shuffled ground-truth baselines on the test set.

To retrain the attention model(s) the generated `flightmare_60.mp4` videos (for training in the same visual environment as the later experiments provide) would have to be provided. Since the space in OSF repositories is limited, we only provide these for the test data. However, we include the trajectory data (`drone.csv`) which can be used to generate the same videos for the training set. For this, see the `generate_data.py` script (under `flightil` in the customized Flightmare repository), particularly the `FlightmareReplicator` class. Then the `train.py` script (under `gazesim`) can be used to train a model.

### End-to-end controllers

Generating test output data for the trained end-to-end controllers can be done using the `test.py` script (under `flightil/dda`). As described in the code repository, this can e.g. be done like this:

```shell
python test.py -rep 10 -mlp controller/checkpoints/encoder-features/model/checkpoint.index -tp controller/inputs/mtt_total_median/flat/test
```

This will let the `encoder-features` model on the test trajectories (with only the first 2 seconds being controlled by the MPC expert). To let the MPC fly the entire trajectory and get model predictions to compute the error, use the `-off` parameter.

Training a model (while generating new data) is as simple as using the `learning_iterative.py` script and provided a `config.yaml` file (see the code repository for an example).

