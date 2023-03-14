package main

import (
	"flag"
	"io"
	"log"
	"os"
	"os/signal"
	"path"
	"runtime"

	"github.com/pkg/profile"

	"hmcalister/hopfield/hopfieldnetwork"
	"hmcalister/hopfield/hopfieldnetwork/datacollector"
	"hmcalister/hopfield/hopfieldnetwork/states"
	"hmcalister/hopfield/hopfieldutils"
)

const DIMENSION = 100
const TARGET_STATES = 100
const LEARNING_RULE = hopfieldnetwork.DeltaLearningRule
const UNITS_UPDATED = 1

var (
	numTrials     *int
	numTestStates *int
	numThreads    *int
	dataDirectory *string
	collector     *datacollector.DataCollector
	logger        *log.Logger
)

func init() {
	numTrials = flag.Int("trials", 1, "The number of trials to undertake.")
	numTestStates = flag.Int("testStates", 1000, "The number of test states to use for each trial.")
	dataDirectory = flag.String("dataDir", "data/trialdata", "The directory to store data files in. Warning: Removes contents of directory!")
	numThreads = flag.Int("threads", 1, "The number of threads to use for relaxation.")
	var logFilePath = flag.String("logFile", "logs/log.txt", "The file to write logs to.")
	flag.Parse()

	logFile, err := os.Create(*logFilePath)
	if err != nil {
		panic("Could not open log file!")
	}
	multiWriter := io.MultiWriter(os.Stdout, logFile)
	logger = log.New(multiWriter, "INFO: ", log.Ldate|log.Ltime|log.Lshortfile)

	log.Printf("Removing data directory %#v\n", *dataDirectory)
	os.RemoveAll(*dataDirectory)
	log.Printf("Creating data directory %#v\n", *dataDirectory)
	os.MkdirAll(*dataDirectory, 0700)

	log.Printf("Creating data collector")
	collector = datacollector.NewDataCollector().
		AddHandler(datacollector.NewTrialEndHandler(path.Join(*dataDirectory, "trialEnd.pq"))).
		AddHandler(datacollector.NewRelaxationResultHandler(path.Join(*dataDirectory, "relaxationResult.pq"))).
		AddHandler(datacollector.NewRelaxationHistoryData(path.Join(*dataDirectory, "relaxationHistory.pq")))
}

// Main method for entry point
func main() {
	defer profile.Start(profile.ClockProfile, profile.ProfilePath("./profiles"), profile.NoShutdownHook).Stop()
	// defer collector.WriteStop()
	go collector.CollectData()

	keyboardInterrupt := make(chan os.Signal, 1)
	signal.Notify(keyboardInterrupt, os.Interrupt)

TrialLoop:
	for trial := 0; trial < *numTrials; trial++ {
		select {
		case <-keyboardInterrupt:
			logger.Printf("RECEIVED KEYBOARD INTERRUPT")
			break TrialLoop
		default:
		}
		logger.Printf("----- TRIAL: %09d -----", trial)
		logger.Printf("Goroutines: %d\n", runtime.NumGoroutine())

		network := hopfieldnetwork.NewHopfieldNetworkBuilder().
			SetNetworkDimension(DIMENSION).
			SetRandMatrixInit(false).
			SetNetworkLearningRule(LEARNING_RULE).
			SetEpochs(100).
			SetMaximumRelaxationIterations(100).
			SetMaximumRelaxationUnstableUnits(0).
			SetUnitsUpdatedPerStep(UNITS_UPDATED).
			SetDataCollector(collector).
			Build()

		stateGenerator := states.NewStateGeneratorBuilder().
			SetRandMin(-1).
			SetRandMax(1).
			SetGeneratorDimension(DIMENSION).
			Build()

		targetStates := stateGenerator.CreateStateCollection(TARGET_STATES)
		network.LearnStates(targetStates)

		testStates := stateGenerator.CreateStateCollection(*numTestStates)
		relaxationResults := network.ConcurrentRelaxStates(testStates, *numThreads)

		trialNumStable := 0
		trialStableStepsTaken := 0
		for stateIndex, result := range relaxationResults {
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
			NumTargetStates:            TARGET_STATES,
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
