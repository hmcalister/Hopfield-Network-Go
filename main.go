package main

import (
	"flag"
	"io"
	"log"
	"os"
	"path"

	"github.com/hmcalister/gonum-matrix-io/pkg/gonumio"
	"github.com/pkg/profile"
	"gonum.org/v1/gonum/mat"

	"hmcalister/hopfield/hopfieldnetwork"
	"hmcalister/hopfield/hopfieldnetwork/datacollector"
	"hmcalister/hopfield/hopfieldnetwork/noiseapplication"
	"hmcalister/hopfield/hopfieldnetwork/states"
	"hmcalister/hopfield/hopfieldutils"
)

const (
	LEARNED_MATRIX_BINARY_SAVE_FILE = "matrix.bin"
	TARGET_STATES_BINARY_SAVE_FILE  = "targetStates.bin"
	PROBE_STATES_BINARY_SAVE_FILE   = "probeStates.bin"
)

var (
	// General network flags

	asymmetricWeightMatrix = flag.Bool("asymmetricWeightMatrix", false, "Allow the weight matrix of the Hopfield network to be asymmetric.")
	networkDimension       = flag.Int("dimension", 100, "The network dimension to simulate.")
	unitsUpdated           = flag.Int("unitsUpdated", 1, "The number of units to update at each step.")

	// Learning rule flags

	learningRuleInt = flag.Int("learningRule", 0, "The learning rule to use.\n0: Hebbian\n1: Delta")
	numEpochs       = flag.Int("epochs", 100, "The number of epochs to train for.")

	// Target and Probe state flags

	numTargetStates        = flag.Int("numTargetStates", 1, "The number of learned states.")
	targetStatesBinaryFile = flag.String("targetStatesFile", "", "Path to the binary file containing the vector collection to use as target states. If present, this method overrides random generation using numTargetStates.")
	numProbeStates         = flag.Int("numProbeStates", 1000, "The number of probe states to use for each trial.")
	probeStatesBinaryFile  = flag.String("probeStatesFile", "", "Path to the binary file containing the vector collection to use as probe states. If present, this method overrides random generation using numProbeStates.")

	// Learning noise flags

	learningNoiseMethodInt = flag.Int("learningNoiseMethod", 0, "The method of applying noise to learned states. Noise scale is determined by the learningNoiseScale Flag.\n0: No Noise\n1: Maximal Inversion\n2:  Random SubMaximal Inversion\n3: Gaussian Application")
	learningNoiseScale     = flag.Float64("learningNoiseScale", 0.0, "The amount of noise to apply to target states during learning.")

	// General program flags

	numThreads    = flag.Int("threads", 1, "The number of threads to use for relaxation.")
	dataDirectory = flag.String("dataDir", "data/trialdata", "The directory to store data files in. Warning: Removes contents of directory!")
	logFilePath   = flag.String("logFile", "logs/log.txt", "The file to write logs to.")
	verbose       = flag.Bool("verbose", false, "Verbose flag to print log messages to stdout.")

	learningRule        hopfieldnetwork.LearningRuleEnum
	learningNoiseMethod noiseapplication.NoiseApplicationEnum
	collector           *datacollector.DataCollector
	logger              *log.Logger
)

func init() {
	// Parse the command line flags and do any mapping from ints (flag variable) to enum (hopfieldnetwork variable)
	flag.Parse()
	learningRule = hopfieldnetwork.LearningRuleEnum(*learningRuleInt)
	learningNoiseMethod = noiseapplication.NoiseApplicationEnum(*learningNoiseMethodInt)

	// Make the directories needed to save data of trials to (if needed)
	os.MkdirAll("logs", 0700)
	os.MkdirAll("profiles", 0700)

	// Tries to open logging file, panics if not possible (since we can't log anything otherwise!)
	logFile, err := os.Create(*logFilePath)
	if err != nil {
		panic("Could not open log file!")
	}

	// Handle verbose flag
	// If set, we make logs point to file *and* stdout
	var multiWriter io.Writer
	if *verbose {
		multiWriter = io.MultiWriter(os.Stdout, logFile)
	} else {
		multiWriter = io.MultiWriter(logFile)
	}
	logger = log.New(multiWriter, "INFO: ", log.Ldate|log.Ltime|log.Lshortfile)

	// Remove old data directory and recreate
	logger.Printf("Creating data directory %#v\n", *dataDirectory)
	os.MkdirAll(*dataDirectory, 0700)

	// Set up data collector to handle events during this trial
	logger.Printf("Creating data collector")
	collector = datacollector.NewDataCollector().
		AddHandler(datacollector.NewRelaxationResultHandler(path.Join(*dataDirectory, "relaxationResult.pq"))).
		AddHandler(datacollector.NewRelaxationHistoryData(path.Join(*dataDirectory, "relaxationHistory.pq"))).
		AddHandler(datacollector.NewTargetStateProbeHandler(path.Join(*dataDirectory, "targetStateProbe.pq")))
}

// Main method for entry point
func main() {
	defer profile.Start(profile.ClockProfile, profile.ProfilePath("./profiles")).Stop()
	go collector.CollectData()

	network := hopfieldnetwork.NewHopfieldNetworkBuilder().
		SetNetworkDimension(*networkDimension).
		SetRandMatrixInit(*asymmetricWeightMatrix).
		SetForceSymmetric(*asymmetricWeightMatrix).
		SetNetworkLearningRule(learningRule).
		SetEpochs(*numEpochs).
		SetMaximumRelaxationIterations(100).
		SetMaximumRelaxationUnstableUnits(0).
		SetLearningNoiseMethod(learningNoiseMethod).
		SetLearningNoiseRatio(*learningNoiseScale).
		SetUnitsUpdatedPerStep(*unitsUpdated).
		SetDataCollector(collector).
		SetLogger(logger).
		Build()

	stateGenerator := states.NewStateGeneratorBuilder().
		SetRandMin(-1).
		SetRandMax(1).
		SetGeneratorDimension(*networkDimension).
		Build()

	// LEARNING PHASE -----------------------------------------------------------------------------
	logger.SetPrefix("Network Learning: ")
	// Make some states and learn them

	var targetStates []*mat.VecDense
	var err error
	if *targetStatesBinaryFile == "" {
		targetStates = stateGenerator.CreateStateCollection(*numTargetStates)
	} else {
		targetStates, err = gonumio.LoadVectorCollection(*targetStatesBinaryFile)
		if err != nil {
			log.Fatalf("ERROR: %v\nTARGET STATES LOADING FAILED", err)
		}
		*numTargetStates = len(targetStates)
	}
	gonumio.SaveVectorCollection(targetStates, path.Join(*dataDirectory, TARGET_STATES_BINARY_SAVE_FILE))
	network.LearnStates(targetStates)
	gonumio.SaveMatrix(network.GetMatrix(), path.Join(*dataDirectory, LEARNED_MATRIX_BINARY_SAVE_FILE))

	// Analyze specifically the learned states and save those results too
	for stateIndex := range targetStates {
		logger.Printf("Analyzing Target State %v\n", stateIndex)
		state := targetStates[stateIndex]
		targetStateData := datacollector.TargetStateProbeData{
			TargetStateIndex: stateIndex,
			IsStable:         network.StateIsStable(state),
			State:            targetStates[stateIndex].RawVector().Data,
			EnergyProfile:    network.AllUnitEnergies(state),
		}
		collector.EventChannel <- hopfieldutils.IndexedWrapper[interface{}]{
			Index: datacollector.DataCollectionEvent_TargetStateProbe,
			Data:  targetStateData,
		}
	}

	// PROBING PHASE ------------------------------------------------------------------------------
	logger.SetPrefix("Network Probing: ")
	// Create and relax a set of probe states

	var probeStates []*mat.VecDense
	if *probeStatesBinaryFile == "" {
		probeStates = stateGenerator.CreateStateCollection(*numProbeStates)
	} else {
		probeStates, err = gonumio.LoadVectorCollection(*probeStatesBinaryFile)
		if err != nil {
			log.Fatalf("ERROR: %v\nPROBE STATES LOADING FAILED", err)
		}
		*numProbeStates = len(probeStates)
	}
	relaxationResults := network.ConcurrentRelaxStates(probeStates, *numThreads)
	logger.Printf("SAVING PROBE STATES")
	gonumio.SaveVectorCollection(probeStates, path.Join(*dataDirectory, PROBE_STATES_BINARY_SAVE_FILE))

	// DATA PROCESSING ----------------------------------------------------------------------------
	logger.SetPrefix("Data Processing: ")
	for stateIndex, result := range relaxationResults {
		logger.Printf("Processing State %v\n", stateIndex)
		event := datacollector.RelaxationResultData{
			StateIndex:         stateIndex,
			Stable:             result.Stable,
			NumSteps:           len(result.StateHistory),
			FinalState:         result.StateHistory[len(result.StateHistory)-1].RawVector().Data,
			DistancesToLearned: result.DistancesToLearned,
			EnergyProfile:      result.EnergyHistory[len(result.EnergyHistory)-1],
		}

		collector.EventChannel <- hopfieldutils.IndexedWrapper[interface{}]{
			Index: datacollector.DataCollectionEvent_RelaxationResult,
			Data:  event,
		}

		for historyIndex, stateHistoryItem := range result.StateHistory {
			historyEvent := datacollector.RelaxationHistoryData{
				StateIndex:    stateIndex,
				HistoryIndex:  historyIndex,
				State:         stateHistoryItem.RawVector().Data,
				EnergyProfile: result.EnergyHistory[historyIndex],
			}

			collector.EventChannel <- hopfieldutils.IndexedWrapper[interface{}]{
				Index: datacollector.DataCollectionEvent_RelaxationHistory,
				Data:  historyEvent,
			}
		}
	}

	// CLEAN UP & FINISH --------------------------------------------------------------------------
	logger.SetPrefix("Clean Up: ")
	// Save to data directory a record of this trial
	networkSummaryData := datacollector.HopfieldNetworkSummaryData{
		NetworkDimension:       *networkDimension,
		LearningRule:           learningRule.String(),
		Epochs:                 *numEpochs,
		LearningNoiseMethod:    learningNoiseMethod.String(),
		LearningNoiseScale:     *learningNoiseScale,
		UnitsUpdated:           *unitsUpdated,
		AsymmetricWeightMatrix: *asymmetricWeightMatrix,
		Threads:                *numThreads,
		TargetStates:           *numTargetStates,
		ProbeStates:            *numProbeStates,
	}
	datacollector.WriteHopfieldNetworkSummary(path.Join(*dataDirectory, "networkSummary.pq"), &networkSummaryData)

	if err := collector.WriteStop(); err != nil {
		logger.Fatalf("ERR: %#v\n", err)
	}
	logger.Printf("Data written successfully")

	logger.Println("DONE")
}
