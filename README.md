# pose_estimation_AWS
this is a pose estimation implement for AWS
![image](https://github.com/bestat/pose_estimation/blob/master/demo.png)

This is the demo which should run on AWS to complete the image deblur task. 

The code basicly from https://github.com/ildoonet/tf-pose-estimation

## Install

### Dependencies

You need dependencies below.

- python3
- tensorflow 1.4.1+
- opencv3, protobuf, python3-tk
- slidingwindow
  - https://github.com/adamrehn/slidingwindow
  - I copied from the above git repo to modify few things.

### Install

Clone the repo and install 3rd-party libraries.

```bash
$ git clone https://www.github.com/bestat/tf-pose-estimation
$ cd tf-pose-estimation
$ pip3 install -r requirements.txt
```

Build c++ library for post processing. See : https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess
```
$ cd tf_pose/pafprocess
$ swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```

### Package Install

Alternatively, you can install this repo as a shared package using pip.

```bash
$ git clone https://www.github.com/ildoonet/tf-pose-estimation
$ cd tf-pose-estimation
$ python setup.py install  # Or, `pip install -e .`
```

# how to use
the same as the face identification task, you can use this command,
```
python3 pose_estimation_on_aws.py --input_path='input.json' --output_path='output.json'
```

then you can get the output.json and frame.jpg.

# input and output format

Input and output image format are all json file. The input json file format is 
```
{
img:data
}
```
here data is the list format of one uint8 image array. For example, [500x500x3] uint8. For further details, please check generate_json_file.py as a reference. I use this file to create the input.json from p1.jpg.

The output json format is the same as the input json file format.


## References

See : [etcs/reference.md](./etcs/reference.md)
