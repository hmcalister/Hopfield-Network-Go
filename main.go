package main

import (
	"flag"
	"log"
	"os"
	"time"

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

	InfoLogger = log.New(logFile, "INFO: ", log.Ldate|log.Ltime|log.Lshortfile)
}

const DOMAIN networkdomain.NetworkDomain = networkdomain.BipolarDomain
const DIMENSION = 100
const TARGET_STATES_RATIO = 0.1

const MIN_UNITS_UPDATED_RATIO = 0.0
const MAX_UNITS_UPDATED_RATIO = 1.0

// Main method for entry point
func main() {
	collector := datacollector.NewDataCollector().
		AddStateRelaxedHandler(*stateLevelDataPath).
		AddOnTrialEndHandler(*trialLevelDataPath)
	defer collector.WriteStop()
	go collector.CollectData()

	srcGenerator := rand.New(rand.NewSource(uint64(time.Now().UnixNano())))
	unitsUpdatedRatioDistSeed := srcGenerator.Uint64()
	unitsUpdatedRatioDist := distuv.Uniform{Min: MIN_UNITS_UPDATED_RATIO, Max: MAX_UNITS_UPDATED_RATIO, Src: rand.NewSource(unitsUpdatedRatioDistSeed)}
	InfoLogger.Printf("unitsUpdatedRatioDist: %#v, Src: %v\n", unitsUpdatedRatioDist, unitsUpdatedRatioDistSeed)

	for trial := 0; trial < *numTrials; trial++ {
		InfoLogger.Printf("----- TRIAL: %09d -----", trial)

		unitsUpdated := int(DIMENSION * unitsUpdatedRatioDist.Rand())
		if unitsUpdated < 1 {
			unitsUpdated = 1
		} else if unitsUpdated > DIMENSION {
			unitsUpdated = DIMENSION
		}

		InfoLogger.Printf("Units Updated: %v\n", unitsUpdated)

		network := hopfieldnetwork.NewHopfieldNetworkBuilder().
			SetNetworkDimension(DIMENSION).
			SetNetworkDomain(DOMAIN).
			SetRandMatrixInit(false).
			SetNetworkLearningRule(hopfieldnetwork.HebbianLearningRule).
			SetEpochs(1).
			SetMaximumRelaxationIterations(100).
			SetMaximumRelaxationUnstableUnits(0).
			SetUnitsUpdatedPerStep(unitsUpdated).
			SetDataCollector(collector).
			Build()

		stateGenerator := states.NewStateGeneratorBuilder().
			SetRandMin(-1).
			SetRandMax(1).
			SetGeneratorDimension(DIMENSION).
			SetGeneratorDomain(DOMAIN).
			Build()

		targetStates := stateGenerator.CreateStateCollection(TARGET_STATES_RATIO * DIMENSION)
		network.LearnStates(targetStates)

		testStates := stateGenerator.CreateStateCollection(*numTestStates)
		relaxationResults := network.ConcurrentRelaxStates(testStates, *numThreads)

		trialNumStable := 0
		trialStableStepsTaken := 0
		for stateIndex, result := range relaxationResults {
			currentTestState := testStates[stateIndex]
			event := datacollector.StateRelaxedData{
				TrialIndex:             trial,
				StateIndex:             stateIndex,
				Stable:                 result.Stable,
				NumSteps:               result.NumSteps,
				FinalState:             currentTestState.RawVector().Data,
				FinalStateEnergyVector: network.AllUnitEnergies(currentTestState),
				DistancesToLearned: hopfieldutils.DistancesToVectorCollection(
					network.GetLearnedStates(),
					currentTestState,
					1.0,
				),
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
			UnitsUpdated:               unitsUpdated,
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
