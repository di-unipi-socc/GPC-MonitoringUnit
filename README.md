# GiòPeopleCounter - Monitoring Unit

A Python Edge script that performs people counting relying on Intel's OpenVINO Toolkit to locally perform live silhouette detection and tracking. The script estimates entrance and exit events through a gate and sends counts to an instance of the [PeopleCounterService
](https://github.com/di-unipi-socc/GPC-PeopleCounterService) Cloud service. More than one MonitoringUnit can be deployed to a building to estimate the overall number of people currently inside it.

The recommended deployment for a MonitoringUnit relies on a RaspberryPi 3B+ equipped with a PiCamera and an Intel Neural Compute Stick 2.

## Modules

The MonitoringUnit is composed of the following folders.

### - Configs 
It contains all configuration parameters.

### - Counts
It implements all classes and functions to perform people counting.

### - Intel
It stores inference models employed for silhouette detection and tracking.

### - Local
It stores the local MonitoringUnit configuration.

### - Logs
It stores log files to keep track of service execution events.

### - mc_tracker
Intel's modules, used to perform inference tasks.

### - Net_io
Implements network interactions, e.g. counts updates and debug frame streaming.

### - Utils
It collects utilities to support main task execution


## Deploy instructions

### 1) RaspberryPI Setup
Install RaspbianOS on a microSD card, and install all tools needed to the code execution:  Python3, OpenVINO Toolkit, PI-Camera Module.

Some useful guides are:
- https://www.raspberrypi.com/documentation/computers/getting-started.html
- https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_raspbian.html
- https://picamera.readthedocs.io/en/release-1.13/quickstart.html

### 2) Load source code
Clone this repository, and fill in config files starting from provided templates. 
The main parameters to configure are:
- Gate Coordinate (local/configs_local.py)
- Inference Engine parameters (configs/mu_conf.py)
- Silhouette reidentification parameters (configs/persons.py)
- Collector Service address/ports/ca_cert.pem (configs/api.py)

### 4) Run:
Execute person_counting.py script to try a single execution.
Change CWD to the project directory, and issue following command:
```bash
python3 person_counting.py
```

