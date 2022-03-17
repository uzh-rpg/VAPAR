# Flightmare (customized version)

For general information on the Flightmare simulator, see the [parent repository](https://github.com/uzh-rpg/flightmare) of this one, as well as the [documentation](https://flightmare.readthedocs.io). This repository contains some changes and additional code used for my Master's thesis (which turned into a paper) and a following project on optical flow.

The main changes to the code from the main repository as of March 14, 2022 are as follows:
- Physics and rendering are completely decoupled, i.e. one can make use of the rendering capabilities of Flightmare without having to use its implemented quadrotor dynamics
- Generation of ground-truth optical flow works in conjunction with the likewise customized [Flightmare Unity rendering engine](https://github.com/swengeler/flightmare_unity)
- Additional code adapts the [Deep Drone Acrobatics](https://github.com/uzh-rpg/deep_drone_acrobatics) framework to work with Flightmare (and a simple MPC in Python, which means that the ROS software stack does not have to be used)

## Publication

The code in this repository represents the main code base for the paper ***"Visual attention prediction improves performance of autonomous drone racing agents"*** published in the PLOS One journal ([link](https://doi.org/10.1371/journal.pone.0264471)).

The data used for the paper can be accessed [here](https://osf.io/uabx4/). The wiki there also contains instructions for replicating the results from the paper.

If you use the code or data in an academic context, please cite the following work:

**APA**:
```
Pfeiffer C, Wengeler S, Loquercio A, Scaramuzza D (2022) Visual attention prediction improves performance of autonomous drone racing agents. PLOS ONE 17(3): e0264471. https://doi.org/10.1371/journal.pone.0264471
```

**Bibtex**:
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

In the paper, we build on [previous work (DDA)](https://doi.org/10.15607/RSS.2020.XVI.040) in which an end-to-end controller network is trained to fly a drone along a reference trajectory using imitation learning. In this work, we use a similar approach but instead of flying (short) acrobatic trajectories, we train controllers to enable flying a long drone racing track. In addition, we investigate the use of human visual attention data (predicted by a model) as an abstraction of visual inputs. 

Thus we compare three main models: a simple baseline model trained on raw **image** data, the model introduced in DDA using **feature tracks**, and our newly introduced model using so-called (attention model) **encoder features**. All of these models are trained by learning to imitate a privileged MPC expert. 

The table below shows the first-person video on an example reference trajectory for the MPC expert and all trained models. Note that for the first two seconds of flight, the MPC always controls the drone. The video for **feature tracks** also shows visualisations of said feature tracks as coloured lines representing the key points tracked across multiple frames. Likewise, the video for (attention) **encoder features** shows the final visual attention predictions of the model as a probability distribution across the image.

| Model | FPV |
| ------------- | ------------- |
| MPC |![MPC FPV visualisation](media/mpc.gif) |
| Images |![Image-based model visualisation](media/images.gif) |
| Feature tracks | ![Feature-track-based model visualisation](media/feature-tracks.gif) |
| Encoder features | ![Attention-encoder-feature-based model visualisation](media/encoder-features.gif) |

<!---
| Model | Details | FPV |
| ------------- | ------------- | ------------- |
| MPC | The drone is flown only by the MPC over the entire track. The first-person video shown on the right is not actually used to determine the commands (only the ground-truth drone state). | ![MPC FPV visualisation](media/mpc.gif) |
| Images | For this model as well as the two below, the first 2s are controlled by the MPC. After that a model trained using raw image data (and other inputs, see the paper) controls the drone. | ![Image-based model visualisation](media/images.gif) |
| Feature tracks | A model using so-called feature tracks as inputs controls the drone. These feature tracks are visualised in the video on the right by colored lines (each representing a key point tracked over multiple frames). The raw image is not used in this model (only for feature track extraction). | ![Feature-track-based model visualisation](media/feature-tracks.gif) |
| Encoder features | A model using so-called encoder features as inputs controls the drone. These features stem from an model predicting human visual attention during drone racing. The final output of that model (an attention distribution) is shown in red on top of the FPV. The raw image is used as input to this model. | ![Attention-encoder-feature-based model visualisation](media/encoder-features.gif) |
-->

<!---
| | |
| --- | --- |
| ![MPC FPV visualisation](media/mpc.gif) | ![Image-based model visualisation](media/images.gif) |
| **MPC** - The drone is flown only by the MPC over the entire track. The first-person video shown above is not actually used to determine the commands (only the ground-truth drone state). | **Images** - For this model as well as the two below, the first 2s are controlled by the MPC. After that a model trained using raw image data (and other inputs, see the paper) controls the drone. |
| ![Feature-track-based model visualisation](media/feature-tracks.gif) | ![Attention-encoder-feature-based model visualisation](media/encoder-features.gif) |
| **Feature tracks** - A model using so-called feature tracks as inputs controls the drone. These feature tracks are visualised in the video on above by colored lines (each representing a key point tracked over multiple frames). The raw image is not used in this model (only for feature track extraction). | **Encoder features** - A model using so-called encoder features as inputs controls the drone. These features stem from an model predicting human visual attention during drone racing. The final output of that model (an attention distribution) is shown in red on top of the FPV. The raw image is used as input to this model. |
-->

## Installation

The dependencies outlined in the original Flightmare documentation should be installed. Installing the Python dependencies is possible using the provided conda `environment.yaml` file. The part that can be a bit trickier is installing the `flightgym` package, which provides an interface in Python for using the Flightmare simulator.

### OpenCV

The main issue with compiling Flightmare that I have found is getting the right OpenCV version as well as a small change that is required in the actual OpenCV source code before it can successfully be used with some of the new code I added (particularly for dealing with receiving images/optical flow data from the Flightmare Unity application). In the following, I try to provide instructions that should *hopefully* work.

First, I have found everything to work best when OpenCV 4 is installed. There are several options to get the development files to compile Flightmare against. The easiest is probably to use the system package manager (which might not be the correct version). The second option is to install `libopencv` using `conda`. The third (and most tedious) option is to clone/download the [OpenCV source code](https://github.com/opencv/opencv).

The next step is to modify one of the files of the OpenCV source code, namely `$OPENCV_INCLUDE_PATH/core/eigen.hpp ` (e.g. `OPENCV_INCLUDE_PATH=/usr/include/opencv4/opencv2` when installed using `apt` on Ubuntu). One of the existing functions simply has to be copied with a slightly different function signature (see [here](https://github.com/opencv/opencv/issues/16606) for more information):

```cpp
// FUNCTION TO BE ADDED
template<typename _Tp>  static inline
void cv2eigen( const Mat& src,
               Eigen::Matrix<_Tp, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dst )
{
    dst.resize(src.rows, src.cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(src.cols, src.rows, traits::Type<_Tp>::value,
             dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        if( src.type() == _dst.type() )
            transpose(src, _dst);
        else if( src.cols == src.rows )
        {
            src.convertTo(_dst, _dst.type());
            transpose(_dst, _dst);
        }
        else
            Mat(src.t()).convertTo(_dst, _dst.type());
    }
    else
    {
        const Mat _dst(src.rows, src.cols, traits::Type<_Tp>::value,
                 dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
    }
}
```

The above is simplest to do with a system-wide installation of OpenCV, which requires admin rights (for installation and modification of the code). As mentioned above, installing OpenCV in a local `conda` environment (from the `conda-forge` channel, since no OpenCV 4 version is available in the default channel) is probably the next-best option and the same steps as above can be followed.

If this is also not possible for some reason, the [manual installation instructions](https://docs.opencv.org/4.5.2/d7/d9f/tutorial_linux_install.html) for OpenCV (for the desired version) can be followed. The source code should still be modified. One issue I found with this method is that unlike what the linked page says, the CMake files required for this project's CMake configuration to find everything are not actually located at `$INSTALL_DIR/cmake/opencv4`, but rather under `$INSTALL_DIR/lib/cmake`. Because of this, I included this path manually in the main `CMakeLists.txt` (namely the line `find_package(OpenCV 4 REQUIRED PATHS /home/simon/.local/lib/cmake)`).

For all of the above cases, **make sure that the output of running CMake (see below) shows that the right OpenCV location has been found**.

### Eigen

While Eigen did not create nearly as many problems as OpenCV, there are two issues that I encountered, that I managed to find a fix for. This is just some information in case similar issues occur despite the implemented fixes.

The first is related to `<unsupported/Eigen/CXX11/Tensor>` apparently not being found during compilation (although this did not always happen, and the files are present...). Since we do not need tensor support for using Flightmare, this can simply be disabled by adding `add_definitions(-DOPENCV_DISABLE_EIGEN_TENSOR_SUPPORT)` to the main `CMakeLists.txt` file (which has already been done for this project).

The second issue is related to receiving large image data from the Flightmare Unity application. This is done (more or less) by receiving it using OpenCV and then converting the data to Eigen. If the images are large (i.e. high image resolution), Eigen might cause crashes, since there is a limit set to how large a matrix can be allocated for this data (at least this is my understanding). The problem can be resolved by raising that limit with `add_definitions(-DEIGEN_STACK_ALLOCATION_LIMIT=3145728)` (as an example value) added to the `CMakeLists.txt` (which is also already done for this project).

### Building the package

Once all of the above is taken care of, the basic steps are as follows. First go to the build directory:

```shell
cd flightlib/build
```

Run CMake to generate build files (the `CMakeLists.txt` file from the parent directory is used):

```shell
cmake ..
```

Compile everything:

```shell
make
```

This should generate (next to all the other files) `flightgym.cpython-39-x86_64-linux-gnu.so`. One aspect to pay attention to here is that the name of this file depends on the Python version that is found by CMake (i.e. the `39` stands for Python 3.9). 

To install the Python package, that file needs to be copied to the correct location and its name is hard-coded in `flightlib/build/setup.py`. Thus, depending on the Python version used to install Flightmare, the name of the file might have to be changed there. 

By default Python 3.9 is assumed to be used, which is **NOT** the one installed by the `environment.yaml` file, because this repository was used more recently by a different project using Python 3.9.

Once all of the above is figured out however, the `flightgym` package can be installed from the build directory with:

```shell
pip install .
```

## Running DDA

As mentioned above, this repository re-implements DDA to work with Flightmare. Here I will explain shortly how to train and test models. Testing in this case means actually flying (test) trajectories rather than predicting control commands offline on unseen data.

### Prerequisites

#### Using the attention model

In order to use the attention model described in the paper (or a gaze prediction model which is also included with the data, see below), another custom Python library needs to be installed.

The [GazeSim](https://github.com/swengeler/gazesim) repository contains code originally used to interface with a previous version of the [Eye Gaze Drone Racing Dataset](https://osf.io/gvdse/), which we use for human gaze/attention data. It also contains code for a variety of attention prediction (and other) models, some of which can be used with the code in this repository for training an end-to-end controller with attention encoder features as inputs.

The only step needed to do this is to clone the aforementioned repository and install the library using:

```shell
pip install .
```

The dependencies for that repository should already be fulfilled by the `environment.yml` file in this repository.

#### Input & output data

The training is performed by loading a trajectory (or multiple) from CSV file(s) and then flying that trajectory (these trajectories) repeatedly. Some pre-processed trajectories from the [Eye Gaze Drone Racing Dataset](https://osf.io/gvdse/) are included in the [data uploaded for the paper](https://osf.io/uabx4/). Specifically, [here](https://mfr.de-1.osf.io/render?url=https://osf.io/tqwdn/?direct%26mode=render%26action=download%26mode=render) is a link to one of the CSV files, which can be taken as an example for the format (note that not all fields are needed). Also included (in the `inputs` folder) are checkpoints for the attention/gaze prediction models used by some of the implemented models that can be trained for DDA.

These trajectories should be specified as absolute paths in the specification YAML file. In contrast, the output data is stored relative to the environment variable `DDA_ROOT`. An additional environment variable that should be set is `FLIGHTMARE_PATH`, which should point to the root of this repository.

#### Flightmare Unity instance

To run either the training or testing script, the Flightmare Unity application of course needs to be running. In addition, the correct ports for communication between the server and client need to be specified (which also allows us to run multiple instances in parallel). That means that the `pub_port` parameter in the specification YAML file (or adjusted manually for the testing script) and the `-input-port` of the Flightmare Unity application need to match. The same needs to be the case for the `sub_port` and `-output-port` parameters.

### Training

All options for training the models should be specified in a YAML file that is used as input for the training script. There is an example template (that trains on a single trajectory) under `flightil/dda/config/template.yaml`. More up-to-date examples for the main models compared for this project are also stored in the same location (`*totmedtraj.yaml`). These use multiple trajectories for training.

To actually run training, go to `flighil/dda` and run:

```shell
python learning_iterative.py --settings_file <path_to_settings_file>
```

### Testing

Once training is done, and the data has been saved under `DDA_ROOT`, the model can be tested by flying trajectories (train or test) by itself, or by letting an MPC fly the trajectory and recording the command predictions. The training script saves a few model checkpoints, the latest one of which should be provided as the model load path for the testing script.

An example use of the testing script would be to go to `flightil/dda` and run:

```shell
python test.py -mlp <path_to_last_model_checkpoint> -tp $DDA_INPUTS_PATH/mtt_total_median/flat/test
```

This runs the specified model on the test trajectories included in `inputs` folder (which can be found in the data published with the paper). For more options of the testing script, see the script itself (or `python test.py -h`). 
