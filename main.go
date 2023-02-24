package main

import (
	"flag"
	"io"
	"log"
	"os"
	"os/signal"
	"time"

	"github.com/pkg/profile"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"

	"hmcalister/hopfield/hopfieldnetwork"
	"hmcalister/hopfield/hopfieldnetwork/datacollector"
	"hmcalister/hopfield/hopfieldnetwork/networkdomain"
	"hmcalister/hopfield/hopfieldnetwork/states"
	"hmcalister/hopfield/hopfieldutils"
)

var (
	numTrials          *int
	numTestStates      *int
	numThreads         *int
	stateLevelDataPath *string
	trialLevelDataPath *string
	InfoLogger         *log.Logger
)

func init() {
	numTrials = flag.Int("trials", 1000, "The number of trials to undertake.")
	numTestStates = flag.Int("testStates", 1000, "The number of test states to use for each trial.")
	stateLevelDataPath = flag.String("stateDataFile", "data/stateData.pq", "The file to write test data about states to. Data is in a parquet format.")
	trialLevelDataPath = flag.String("trialDataFile", "data/trialData.pq", "The file to write test data about trials to. Data is in a parquet format.")
	numThreads = flag.Int("threads", 1, "The number of threads to use for relaxation.")
	var logFilePath = flag.String("logFile", "logs/log.txt", "The file to write logs to.")
	flag.Parse()

	logFile, err := os.Create(*logFilePath)
	if err != nil {
		panic("Could not open log file!")
	}

	multiWriter := io.MultiWriter(os.Stdout, logFile)
	InfoLogger = log.New(multiWriter, "INFO: ", log.Ldate|log.Ltime|log.Lshortfile)
}

const DOMAIN networkdomain.NetworkDomain = networkdomain.BipolarDomain
const DIMENSION = 100
const TARGET_STATES_RATIO_MIN = 0.0
const TARGET_STATES_RATIO_MAX = 0.5
const UNITS_UPDATED = 5

// Main method for entry point
func main() {
	defer profile.Start(profile.ProfilePath("./profiles")).Stop()

	collector := datacollector.NewDataCollector().
		AddStateRelaxedHandler(*stateLevelDataPath).
		AddOnTrialEndHandler(*trialLevelDataPath)
	defer collector.WriteStop()
	go collector.CollectData()

	keyboardInterrupt := make(chan os.Signal, 1)
	signal.Notify(keyboardInterrupt, os.Interrupt)

	srcGenerator := rand.New(rand.NewSource(uint64(time.Now().UnixNano())))
	targetStateRatioDistSeed := srcGenerator.Uint64()
	targetStateRatioDist := distuv.Uniform{Min: TARGET_STATES_RATIO_MIN, Max: TARGET_STATES_RATIO_MAX, Src: rand.NewSource(targetStateRatioDistSeed)}
	InfoLogger.Printf("targetStateRatioDist: %#v, Src: %v\n", targetStateRatioDist, targetStateRatioDistSeed)

TrialLoop:
	for trial := 0; trial < *numTrials; trial++ {
		select {
		case <-keyboardInterrupt:
			InfoLogger.Printf("RECEIVED KEYBOARD INTERRUPT")
			break TrialLoop
		default:
		}
		InfoLogger.Printf("----- TRIAL: %09d -----", trial)

		numTargetStates := int(DIMENSION * targetStateRatioDist.Rand())
		if numTargetStates < 1 {
			numTargetStates = 1
		}

		InfoLogger.Printf("Target States: %v\n", numTargetStates)

		network := hopfieldnetwork.NewHopfieldNetworkBuilder().
			SetNetworkDimension(DIMENSION).
			SetNetworkDomain(DOMAIN).
			SetRandMatrixInit(false).
			SetNetworkLearningRule(hopfieldnetwork.DeltaLearningRule).
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
			SetGeneratorDomain(DOMAIN).
			Build()

		targetStates := stateGenerator.CreateStateCollection(numTargetStates)
		network.LearnStates(targetStates)

		testStates := stateGenerator.CreateStateCollection(*numTestStates)
		relaxationResults := network.ConcurrentRelaxStates(testStates, *numThreads)

		trialNumStable := 0
		trialStableStepsTaken := 0
		for stateIndex, result := range relaxationResults {
			// currentTestState := testStates[stateIndex]
			event := datacollector.StateRelaxedData{
				TrialIndex: trial,
				StateIndex: stateIndex,
				Stable:     result.Stable,
				NumSteps:   result.NumSteps,
			}

			collector.EventChannel <- hopfieldutils.IndexedWrapper[interface{}]{
				Index: datacollector.DataCollectionEvent_OnStateRelax,
				Data:  event,
			}

			if result.Stable {
				trialNumStable += 1
				trialStableStepsTaken += result.NumSteps
			}
		}
		trialResult := datacollector.OnTrialEndData{
			TrialIndex:                 trial,
			NumTargetStates:            numTargetStates,
			NumberStableStates:         trialNumStable,
			StableStatesMeanStepsTaken: float64(trialStableStepsTaken) / float64(trialNumStable),
		}
		collector.EventChannel <- hopfieldutils.IndexedWrapper[interface{}]{
			Index: datacollector.DataCollectionEvent_OnTrialEnd,
			Data:  trialResult,
		}
	}

	collector.WriteStop()
}
