# Hopfield Network Go
#### Author: Hayden McAlister

![Badge for the Go build+test workflow](https://github.com/hmcalister/Hopfield-Network-Go/actions/workflows/go.yml/badge.svg?branch=main)

## Running the Project

First, compile the project (to avoid potential slow downs from using `go run`):
- `go build .`

Then, run the resulting binary:
- `./hopfield`

Note that much of the functionality of the network is determined by command line arguments given at run time. Use `./hopfield -h` to see a list of these.

Data on the run is saved to the directory specified (default: `data/trialdata`), which consists of a collection of parquet files pertaining to different sections of the hopfield networks behavior. See the section on [Data Files](#data-files)

## Data Files

### `networkSummary.pq`

Defines data on the network. Independent of epochs, target states, probes, etc...

Effectively meta data on the trial.

#### Fields
- `NetworkDimension`
    - The dimension of the network. Integer.
- `LearningRule`
    - The learning rule used. String.
- `Epochs`
    - The maximum number of epochs training is allowed to go on for. Integer.
- `LearningNoiseMethod`
    - The method used to apply noise to states during learning. String.
- `LearningNoiseScale`
    - The scale used with the learning noise method. For inversion methods, this indicates how many units can be flipped. For gaussian methods, this is the standard deviation. Float.
- `UnitsUpdated`
    - The number of units updated at each step during relaxation. Integer.
- `AsymmetricWeightMatrix`
    - Flag to indicate if the weight matrix is forced to be symmetric. Boolean.
- `Threads`
    - The number of threads used to relax states. Integer.
- `TargetStates`
    - The number of target states used in learning. Integer.
- `ProbeStates`
    - The number of probe states used in probing. Integer.

### `learnStateData.pq`

Collects data on the learning behavior of the network. In particular, measures properties of target states during the epochs of learning. Measured for every target state, for every epoch of training.

#### Fields

- `Epoch`
    - The learning epoch this instance relates to. Integer.
- `TargetStateIndex`
    - The target state this instances relates to. Integer.
- `EnergyProfile`
    - The energy profile of this instance *before* this epoch is applied. []float64.
- `Stable`
    - A flag to represent if this target state is now stable in the network. Bool.

### `targetStateProbe.pq`

Collects data on the target states after training. Measured after the network has trained in full.

#### Fields

- `TargetStateIndex`
    - The target state this instances relates to. Integer.
- `IsStable`
    - Flag to specify if this state is stable with respect to the trained network, determined by the energy profile. Bool.
- `State`
    - The actual target state associated with this target state index. []float64
- `EnergyProfile`
    - The energy profile of the target state with respect to the trained network. []float64

### `relaxationResult.pq`

Collects data on the results of relaxing probe states. Note this only involves the *results* and does not collect data on any intermediate steps. See `RelaxationHistory.pq` for this.

#### Fields
- `StateIndex`
    - The probe state index this instance refers to. Integer.
- `Stable`
    - Flag to indicate if this state relaxed to a stable state. Boolean
- `NumSteps`
    - The number of steps required to relax to the final state. Integer.
- `FinalState`
    - The vector representing the final state this probe state mapped on to. []float64.
- `DistancesToTargets`
    - A vector representing the distances (Manhattan distance) to each target state. Note the index into this vector corresponds to `TargetStateIndex`. []float64.
- `EnergyProfile`
    - A vector representing the energy profile of the final state with respect to the trained network. []float64.

### `RelaxationHistory.pq`

Collects data on the relaxing probe states *during* relaxation. This involves a lot of data!

#### Fields

- `StateIndex`
    - The probe state index this instance refers to. Integer.
- `StepIndex`
    - The index of the step this instance refers to. Integer.
- `State`
    - The state of the probe at this instance. []float64.
- `EnergyProfile`
    - The energy profile of the state at this step. []float64.

### `matrix.bin`

A binary representation of the weight matrix after training.

### `targetStates.bin`

A binary file consisting of a matrix. Each row in this matrix is a different target state for this trial.

## Introduction

This project is an investigation into implementing the Hopfield network (and some other supporting methods) in Go using gonum as a linear algebra backend. This project is intended to be clean and extensible, as well as blazing fast and scalable with CPU cores via threading. 

In future it may be interesting to try and port this project to use a different backend project - one that leverages linear algebra on CUDA to scale instead with the GPU.


## Why Go?

Go was chosen for this project for the following reasons:

- It was found to be fast (see the profiling and testing [in this repository](https://github.com/hmcalister/Linear-Algebra-Profiling) - be sure to checkout the dashboard!)

- Tensorflow was found to scale much better by leveraging the GPU, but ensuring the code continued to scale required awkward vectorized methods that were prone to bugs.

- Rust was found to scale nearly as well as Go on the CPU, and has nicer memory safety. However, multithreading proved to be difficult, and the implementation did not continue very far past the initial experiments. Check out the [Rust implementation](https://github.com/hmcalister/Hopfield-Network-Rust).

- Go was found to scale very slightly better on the CPU, and after the [initial implementation](https://github.com/hmcalister/Hopfield-Network-Go) the language was found to be a nicer fit. Higher velocity development wins the day!
