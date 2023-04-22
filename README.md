# ORB_SLAM3-PythonBindings

A python wrapper for ORB_SLAM3, which can be found at [ORB_SLAM3](https://github.com/bjr12l/ORB_SLAM3).
This is designed to work with the base version of ORB_SLAM3, with a couple of minimal API changes to access the system output.

## Add python 3.8 Support

It has been tested on Ubuntu 20.04

## Installation

### Prerequesities:
- Docker

### Docker:
1. Build ORB_SLAM3 base image by running `docker compose build dev` from [ORB_SLAM3 repo](https://github.com/bjr12l/ORB_SLAM3)
2. Build image here by running `docker compose build dev`

## Running:
- You need to have vocabluary from the [ORB_SLAM3 repo](https://github.com/bjr12l/ORB_SLAM3)
- And a configuration file with camera parameters. See [this for example](examples/k2/k2.yaml)


You can try with any of your own videos here is the key code:
```python
import orbslam3

slam = orbslam3.System(path_to_vocab, path_to_config, orbslam3.Sensor.MONOCULAR)
slam.set_use_viewer(True)
slam.initialize()

ret, frame = vid.read()
slam.process_image_mono(frame, vid.get(cv2.CAP_PROP_POS_MSEC) / 1000, "")

```


## License

This code is licensed under the BSD Simplified license, although it requires and links to ORB_SLAM3, which is available under the GPLv3 license

It uses pyboostcvconverter (https://github.com/Algomorph/pyboostcvconverter) by Gregory Kramida under the MIT licence (see pyboostcvconverter-LICENSE).
