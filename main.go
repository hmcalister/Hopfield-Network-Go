package main

import (
	"flag"
	"io"
	"log"
	"os"
	"path"
	"runtime"

	"github.com/pkg/profile"

	"hmcalister/hopfield/hopfieldnetwork"
	"hmcalister/hopfield/hopfieldnetwork/datacollector"
	"hmcalister/hopfield/hopfieldnetwork/states"
	"hmcalister/hopfield/hopfieldutils"
)

var (
	numTrials          = flag.Int("trials", 1, "The number of trials to undertake.")
	numThreads         = flag.Int("threads", 1, "The number of threads to use for relaxation.")
	networkDimension   = flag.Int("dimension", 100, "The network dimension to simulate.")
	learningRuleInt    = flag.Int("learningRule", 0, "The learning rule to use.\n0: Hebbian\n1: Delta")
	numEpochs          = flag.Int("epochs", 100, "The number of epochs to train for.")
	numTargetStates    = flag.Int("targetStates", 1, "The number of learned states.")
	numTestStates      = flag.Int("testStates", 1000, "The number of test states to use for each trial.")
	learningNoiseRatio = flag.Float64("learningNoiseRatio", 0.0, "The amount of noise to apply to target states during learning.")
	unitsUpdated       = flag.Int("unitsUpdated", 1, "The number of units to update at each step.")
	dataDirectory      = flag.String("dataDir", "data/trialdata", "The directory to store data files in. Warning: Removes contents of directory!")
	logFilePath        = flag.String("logFile", "logs/log.txt", "The file to write logs to.")
	verbose            = flag.Bool("verbose", false, "Verbose flag to print log messages to stdout.")

	learningRule hopfieldnetwork.LearningRuleEnum
	collector    *datacollector.DataCollector
	logger       *log.Logger
)

func init() {
	flag.Parse()
	learningRule = hopfieldnetwork.LearningRuleEnum(*learningRuleInt)

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
		NetworkDimension:   *networkDimension,
		LearningRule:       learningRule.String(),
		Epochs:             *numEpochs,
		LearningNoiseRatio: *learningNoiseRatio,
		UnitsUpdated:       *unitsUpdated,
		Trials:             *numTrials,
		Threads:            *numThreads,
		TargetStates:       *numTargetStates,
		TestStates:         *numTestStates,
	}
	datacollector.WriteHopfieldNetworkSummary(path.Join(*dataDirectory, "networkSummary.pq"), &networkSummaryData)

	logger.Printf("Creating data collector")
	collector = datacollector.NewDataCollector().
		AddHandler(datacollector.NewTrialEndHandler(path.Join(*dataDirectory, "trialEnd.pq"))).
		AddHandler(datacollector.NewRelaxationResultHandler(path.Join(*dataDirectory, "relaxationResult.pq"))).
		AddHandler(datacollector.NewRelaxationHistoryData(path.Join(*dataDirectory, "relaxationHistory.pq")))
}

// Main method for entry point
func main() {
	defer profile.Start(profile.ClockProfile, profile.ProfilePath("./profiles")).Stop()
	// defer collector.WriteStop()
	go collector.CollectData()

	for trial := 0; trial < *numTrials; trial++ {
		logger.SetPrefix("INFO: ")
		logger.Printf("----- TRIAL: %09d -----", trial)
		logger.Printf("Goroutines: %d\n", runtime.NumGoroutine())

		network := hopfieldnetwork.NewHopfieldNetworkBuilder().
			SetNetworkDimension(*networkDimension).
			SetRandMatrixInit(false).
			SetNetworkLearningRule(learningRule).
			SetEpochs(*numEpochs).
			SetMaximumRelaxationIterations(100).
			SetMaximumRelaxationUnstableUnits(0).
			SetLearningNoiseRatio(*learningNoiseRatio).
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

		logger.SetPrefix("Network Testing: ")
		testStates := stateGenerator.CreateStateCollection(*numTestStates)
		relaxationResults := network.ConcurrentRelaxStates(testStates, *numThreads)

		logger.SetPrefix("Data Processing: ")
		trialNumStable := 0
		trialStableStepsTaken := 0
		for stateIndex, result := range relaxationResults {
			logger.Printf("Processing State %v\n", stateIndex)
			event := datacollector.RelaxationResultData{
				TrialIndex:         trial,
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
					TrialIndex:    trial,
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
		trialResult := datacollector.TrialEndData{
			TrialIndex:                 trial,
			NumTestStates:              *numTestStates,
			NumTargetStates:            *numTargetStates,
			NumStableStates:            trialNumStable,
			StableStatesMeanStepsTaken: float64(trialStableStepsTaken) / float64(trialNumStable),
		}
		collector.EventChannel <- hopfieldutils.IndexedWrapper[interface{}]{
			Index: datacollector.DataCollectionEvent_TrialEnd,
			Data:  trialResult,
		}
		logger.Printf("Stable States: %05d/%05d\n", trialNumStable, *numTestStates)
	} //end Trial Loop

	if err := collector.WriteStop(); err != nil {
		logger.Fatalf("ERR: %#v\n", err)
	}
}
