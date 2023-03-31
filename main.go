package main

import (
	"flag"
	"io"
	"log"
	"os"
	"path"

	"github.com/pkg/profile"

	"hmcalister/hopfield/hopfieldnetwork"
	"hmcalister/hopfield/hopfieldnetwork/datacollector"
	"hmcalister/hopfield/hopfieldnetwork/noiseapplication"
	"hmcalister/hopfield/hopfieldnetwork/states"
	"hmcalister/hopfield/hopfieldutils"
)

var (
	numThreads             = flag.Int("threads", 1, "The number of threads to use for relaxation.")
	networkDimension       = flag.Int("dimension", 100, "The network dimension to simulate.")
	learningRuleInt        = flag.Int("learningRule", 0, "The learning rule to use.\n0: Hebbian\n1: Delta")
	numEpochs              = flag.Int("epochs", 100, "The number of epochs to train for.")
	numTargetStates        = flag.Int("targetStates", 1, "The number of learned states.")
	numTestStates          = flag.Int("testStates", 1000, "The number of test states to use for each trial.")
	learningNoiseMethodInt = flag.Int("learningNoiseMethod", 0, "The method of applying noise to learned states. Noise scale is determined by the learningNoiseScale Flag.\n0: No Noise\n1: Maximal Inversion\n2:  Random SubMaximal Inversion\n3: Gaussian Application")
	learningNoiseScale     = flag.Float64("learningNoiseScale", 0.0, "The amount of noise to apply to target states during learning.")
	asymmetricWeightMatrix = flag.Bool("asymmetricWeightMatrix", false, "Allow the weight matrix of the Hopfield network to be asymmetric.")
	unitsUpdated           = flag.Int("unitsUpdated", 1, "The number of units to update at each step.")
	dataDirectory          = flag.String("dataDir", "data/trialdata", "The directory to store data files in. Warning: Removes contents of directory!")
	logFilePath            = flag.String("logFile", "logs/log.txt", "The file to write logs to.")
	verbose                = flag.Bool("verbose", false, "Verbose flag to print log messages to stdout.")

	learningRule        hopfieldnetwork.LearningRuleEnum
	learningNoiseMethod noiseapplication.NoiseApplicationEnum
	collector           *datacollector.DataCollector
	logger              *log.Logger
)

func init() {
	flag.Parse()
	learningRule = hopfieldnetwork.LearningRuleEnum(*learningRuleInt)
	learningNoiseMethod = noiseapplication.NoiseApplicationEnum(*learningNoiseMethodInt)

	os.MkdirAll("logs", 0700)
	os.MkdirAll("profiles", 0700)
	logFile, err := os.Create(*logFilePath)
	if err != nil {
		panic("Could not open log file!")
	}

	var multiWriter io.Writer
	if *verbose {
		multiWriter = io.MultiWriter(os.Stdout, logFile)
	} else {
		multiWriter = io.MultiWriter(logFile)
	}
	logger = log.New(multiWriter, "INFO: ", log.Ldate|log.Ltime|log.Lshortfile)

	logger.Printf("Removing data directory %#v\n", *dataDirectory)
	os.RemoveAll(*dataDirectory)
	logger.Printf("Creating data directory %#v\n", *dataDirectory)
	os.MkdirAll(*dataDirectory, 0700)

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
		TestStates:             *numTestStates,
	}
	datacollector.WriteHopfieldNetworkSummary(path.Join(*dataDirectory, "networkSummary.pq"), &networkSummaryData)

	logger.Printf("Creating data collector")
	collector = datacollector.NewDataCollector().
		AddHandler(datacollector.NewStateAggregateHandler(path.Join(*dataDirectory, "stateAggregate.pq"))).
		AddHandler(datacollector.NewRelaxationResultHandler(path.Join(*dataDirectory, "relaxationResult.pq"))).
		AddHandler(datacollector.NewRelaxationHistoryData(path.Join(*dataDirectory, "relaxationHistory.pq"))).
		AddHandler(datacollector.NewTargetStateProbeHandler(path.Join(*dataDirectory, "targetStateProbe.pq")))
}

// Main method for entry point
func main() {
	defer profile.Start(profile.ClockProfile, profile.ProfilePath("./profiles")).Stop()
	// defer collector.WriteStop()
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

	logger.SetPrefix("Network Learning: ")
	targetStates := stateGenerator.CreateStateCollection(*numTargetStates)
	network.LearnStates(targetStates)
	if err := hopfieldutils.SaveMatrix(network.GetMatrix(), path.Join(*dataDirectory, "networkMatrix")); err != nil {
		log.Printf("Error '%v' while saving Hopfield weights", err)
	}

	for stateIndex := range targetStates {
		logger.Printf("Analyzing Target State %v\n", stateIndex)
		state := targetStates[stateIndex]
		targetStateData := datacollector.TargetStateProbeData{
			TargetStateIndex: stateIndex,
			IsStable:         network.StateIsStable(state),
			EnergyProfile:    network.AllUnitEnergies(state),
		}
		collector.EventChannel <- hopfieldutils.IndexedWrapper[interface{}]{
			Index: datacollector.DataCollectionEvent_TargetStateProbe,
			Data:  targetStateData,
		}
	}

	logger.SetPrefix("Network Testing: ")
	testStates := stateGenerator.CreateStateCollection(*numTestStates)
	relaxationResults := network.ConcurrentRelaxStates(testStates, *numThreads)

	logger.SetPrefix("Data Processing: ")
	trialNumStable := 0
	trialStableStepsTaken := 0
	for stateIndex, result := range relaxationResults {
		logger.Printf("Processing State %v\n", stateIndex)
		event := datacollector.RelaxationResultData{
			StateIndex:         stateIndex,
			Stable:             result.Stable,
			NumSteps:           len(result.StateHistory),
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

		if result.Stable {
			trialNumStable += 1
			trialStableStepsTaken += len(result.StateHistory)
		}
	}
	trialResult := datacollector.StateAggregateData{
		NumTestStates:              *numTestStates,
		NumTargetStates:            *numTargetStates,
		NumStableStates:            trialNumStable,
		StableStatesMeanStepsTaken: float64(trialStableStepsTaken) / float64(trialNumStable),
	}
	collector.EventChannel <- hopfieldutils.IndexedWrapper[interface{}]{
		Index: datacollector.DataCollectionEvent_StateAggregate,
		Data:  trialResult,
	}
	logger.Printf("Stable States: %05d/%05d\n", trialNumStable, *numTestStates)

	if err := collector.WriteStop(); err != nil {
		logger.Fatalf("ERR: %#v\n", err)
	}
}
