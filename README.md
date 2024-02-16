# Objective
Provide a system that allows people to develop performant models without coding.

# Get started
## Installation
The first step is to clone the repository.
```sh
git@github.com:manazhao/CodelessML.git
```

After cloing the repository, enter the project and run all the test targets to
ensure the code is healthy.
```
./bazel test ...
```
Note: the test may take a while to finish as some model training and eval
related targets take more time to finish.

## Run example code
The examples are located under the [examples](codeless_ml/ml/examples) folder.
Each example is a [protobuf](https://github.com/protocolbuffers/protobuf)
formatted text file. The artifacts, e.g., pretrained models and datastes, used
by the examples are under the [artifacts](codeless_ml/ml/artifacts) folder. The
following command launchs training and evaluating a portuguese-to-english
transformer based translation model.

```sh
./bazel run codeless_ml/ml:train  -- \
--trainer_config_file=codeless_ml/ml/examples/machine_translation.prototext \
--job=train --alsologtostderr
```
