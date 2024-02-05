# Objective
**Mission**: Provide a system that allows people to develop performant models
without coding.

# Get started
## Installation
The first step is to clone the repository.
```sh
git@github.com:manazhao/CodelessML.git
```

## Run example code
The examples are located under the [examples](codeless_ml/ml/examples) folder.
Each example is a [protobuf]() formatted text file. The artifacts, e.g.,
pretrained models and datastes, used by the examples are under the [artifacts]()
folder. 

```sh
./bazel run codeless_ml/ml:train  -- \
--trainer_config_file=codeless_ml/ml/examples/machine_translation.prototext \
--job=train --alsologtostderr
```


