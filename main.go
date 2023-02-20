package main

import (
	"flag"
	"log"
	"os"

	"gonum.org/v1/gonum/stat/distuv"

	"hmcalister/hopfield/hopfieldnetwork"
	"hmcalister/hopfield/hopfieldnetwork/networkdomain"
	"hmcalister/hopfield/hopfieldnetwork/states"
	"hmcalister/hopfield/hopfieldutils"
)

var (
	numTrials     *int
	numTestStates *int
	dataFilePath  *string
	InfoLogger    *log.Logger
	ErrorLogger   *log.Logger
)

type DataEntry struct {
	Dimension            int     `parquet:"name=dimension, type=INT32"`
	TargetStates         int     `parquet:"name=targetStates, type=INT32"`
	UnitsUpdated         int     `parquet:"name=unitsUpdated, type=INT32"`
	TestStates           int     `parquet:"name=testStates, type=INT32"`
	NumStable            int     `parquet:"name=stableStates, type=INT32"`
	MeanStableStepsTaken float64 `parquet:"name=meanStableStepsTaken, type=FLOAT"`
}

func init() {
	numTrials = flag.Int("trials", 1000, "The number of trials to undertake.")
	numTestStates = flag.Int("testStates", 1000, "The number of test states to use for each trial.")
	dataFilePath = flag.String("dataFile", "data/data.pq", "The file to write test data to. Data is in a parquet format.")
	var logFilePath = flag.String("logFile", "logs/log.txt", "The file to write logs to.")
	flag.Parse()

	logFile, err := os.Create(*logFilePath)
	if err != nil {
		panic("Could not open log file!")
	}

	InfoLogger = log.New(logFile, "INFO: ", log.Ldate|log.Ltime|log.Lshortfile)
	ErrorLogger = log.New(logFile, "Error: ", log.Ldate|log.Ltime|log.Lshortfile)
}

const DOMAIN networkdomain.NetworkDomain = networkdomain.BipolarDomain

const MIN_DIMENSION = 50
const MAX_DIMENSION = 100

const MIN_TARGET_STATES_RATIO = 0.05
const MAX_TARGET_STATES_RATIO = 1.0

const MIN_UNITS_UPDATED_RATIO = 0.0
const MAX_UNITS_UPDATED_RATIO = 1.0

// Main method for entry point
func main() {
	dataWriter := hopfieldutils.ParquetWriter(*dataFilePath, new(DataEntry))
	defer dataWriter.WriteStop()

	dimensionDist := distuv.Uniform{Min: MIN_DIMENSION, Max: MAX_DIMENSION}
	targetStatesRatioDist := distuv.Uniform{Min: MIN_TARGET_STATES_RATIO, Max: MAX_TARGET_STATES_RATIO}
	unitsUpdatedRatioDist := distuv.Uniform{Min: MIN_UNITS_UPDATED_RATIO, Max: MAX_UNITS_UPDATED_RATIO}
	for trial := 0; trial < *numTrials; trial++ {
		InfoLogger.Printf("----- TRIAL: %09d -----", trial)

		floatDimension := dimensionDist.Rand()
		dimension := int(floatDimension)

		numTargetStates := int(floatDimension * targetStatesRatioDist.Rand())
		if numTargetStates < 0 {
			numTargetStates = 0
		} else if numTargetStates > dimension {
			numTargetStates = dimension
		}

		unitsUpdated := int(floatDimension * unitsUpdatedRatioDist.Rand())
		if unitsUpdated < 1 {
			unitsUpdated = 1
		} else if unitsUpdated > dimension {
			unitsUpdated = dimension
		}

		InfoLogger.Printf("Dimension: %v\n", dimension)
		InfoLogger.Printf("Num Target States: %v\n", numTargetStates)
		InfoLogger.Printf("Units Updated: %v\n", unitsUpdated)
		InfoLogger.Printf("Test States: %v\n", *numTestStates)

		network := hopfieldnetwork.NewHopfieldNetworkBuilder().
			SetNetworkDimension(dimension).
			SetNetworkDomain(DOMAIN).
			SetRandMatrixInit(false).
			SetNetworkLearningRule(hopfieldnetwork.HebbianLearningRule).
			SetEpochs(1).
			SetMaximumRelaxationIterations(100).
			SetMaximumRelaxationUnstableUnits(0).
			SetUnitsUpdatedPerStep(unitsUpdated).
			Build()

		stateGenerator := states.NewStateGeneratorBuilder().
			SetRandMin(-1).
			SetRandMax(1).
			SetGeneratorDimension(dimension).
			SetGeneratorDomain(DOMAIN).
			Build()

		targetStates := stateGenerator.CreateStateCollection(numTargetStates)
		network.LearnStates(targetStates)

		testStates := stateGenerator.CreateStateCollection(*numTestStates)
		testResults := network.ConcurrentRelaxStates(testStates, 8)

		numStable := 0
		numStepsTotal := 0
		for _, result := range testResults {
			if result.Stable {
				numStable += 1
				numStepsTotal += result.NumSteps
			}
		}

		numStepsAvg := float64(numStepsTotal) / float64(numStable)

		InfoLogger.Printf("Stable Test States: %v\n", numStable)
		InfoLogger.Printf("Mean Stable States Steps Taken: %v\n", numStepsAvg)

		dataWriter.Write(DataEntry{
			Dimension:            dimension,
			TargetStates:         numTargetStates,
			UnitsUpdated:         unitsUpdated,
			TestStates:           *numTestStates,
			NumStable:            numStable,
			MeanStableStepsTaken: numStepsAvg,
		})
	}
}
