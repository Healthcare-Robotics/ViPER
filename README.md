# Visual Pressure Estimation for Robots (ViPER)

[Paper](https://arxiv.org/abs/2303.07344) | [Video](https://youtu.be/z5Rttv1oZJA) | [Dataset/Models](https://1drv.ms/f/s!AjebifpxoPl5hO0YUiva8M0ClODhTw?e=TFBP02)

ViPER is a neural network that visually estimates contact pressure given an RGB image from an eye-in-hand camera. ViPER enables precision grasping of small objects (e.g. a paperclip) on unseen cluttered surfaces.

![alt text](https://github.com/Healthcare-Robotics/ViPER/blob/main/images/viper_headliner.png "Visual Pressure Estimation for Robots")

# Setup (Tested in Ubuntu 20.04)
- Clone this repo, as well as [this one](https://github.com/qubvel/segmentation_models.pytorch), and place them in the same directory.
- Install the requirements in requirements.txt.
- Install remaining dependencies: `pip install open3d trimesh shapely mapbox_earcut`
- Install the Sensel API from the deb file in [this repo](https://github.com/sensel/sensel-api).
- Download the dataset and models [here](https://1drv.ms/f/s!AjebifpxoPl5hO0YUiva8M0ClODhTw?e=TFBP02).
- Place folders in the dataset in `/data` (if training) and the folders in the models folder in `/checkpoints`.

## Collecting data
There are 4 types of data to capture:
- train: Images that are paired with both pressure ground truth and force/torque ground truth
- test: Same as train, but for evaluating the network on predicting pressure
- weak_train: Images that are paired with only force/torque ground truth
- weak_test: Same as train, but for evaluating the network on predicting only force/torque

### Running the data collection script
- `python -m recording.capture_data --config <yaml file name from /config> --view --robot_state --stage <train, test, weak_train, weak_test> --folder <pick a name>`
- see keyboard_teleop() in `robot/robot_control.py` for keyboard controls.

## Training
- The training script will save a folder containing models from the same run. The folder is named `<config>_<index>` and the models are named `model_<epoch>`.
- Run the training script: `python -m prediction.trainer_weak --config <yaml file name from /config>`

## Robot (Hello Robot Stretch)
To run on a Hello Robot Stretch, clone this repo on the robot. Then, verify the robot and PC are on the same network and that the IPs match those in `/robot/zmq_client.py`
Run `python -m robot.robot_control`

## Live model
- To run a live model, ensure that checkpoints are saved in a format resembling `checkpoints/<config>_<index>/model_<epoch>.pth`. Then run:
- Run the live model: `python -m prediction.live_model --config <yaml file name from /config> --view --index <index> --epoch <epoch>`
(You do not need to specify index and epoch if using the provided models)

## Demo
- `python -m demo.grasp_demo --config <yaml file name from /config> --view --index <index> --epoch <epoch>`

## Citing this work
If you use ViPER for academic research, please cite our paper. Below is the BibTeX entry for citation:
```bibtex
@inproceedings{collins2023visual,
  title={Visual Contact Pressure Estimation for Grippers in the Wild},
  author={Collins, Jeremy A and Houff, Cody and Grady, Patrick and Kemp, Charles C},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2023}
  organization={IEEE}
}
```