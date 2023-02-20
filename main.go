package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"

	"gonum.org/v1/gonum/stat/distuv"

	"hmcalister/hopfield/hopfieldnetwork"
	"hmcalister/hopfield/hopfieldnetwork/networkdomain"
	"hmcalister/hopfield/hopfieldnetwork/states"
)

var (
	numTrials     *int
	numTestStates *int
	dataFilePath  *string
	InfoLogger    *log.Logger
	ErrorLogger   *log.Logger
)

func init() {
	numTrials = flag.Int("trials", 1000, "The number of trials to undertake.")
	numTestStates = flag.Int("testStates", 1000, "The number of test states to use for each trial.")
	dataFilePath = flag.String("dataFile", "data/data.csv", "The file to write test data to. Data is in a CSV format.")
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
	datafile, _ := os.Create(*dataFilePath)
	defer datafile.Close()

	datawriter := csv.NewWriter(datafile)
	defer datawriter.Flush()

	datawriter.Write([]string{"Dimension", "Target States", "Units Activated", "Test States", "Stable Test States"})

	dimensionDist := distuv.Uniform{Min: MIN_DIMENSION, Max: MAX_DIMENSION}
	targetStatesRatioDist := distuv.Uniform{Min: MIN_TARGET_STATES_RATIO, Max: MAX_TARGET_STATES_RATIO}
	unitsUpdatedRatioDist := distuv.Uniform{Min: MIN_UNITS_UPDATED_RATIO, Max: MAX_UNITS_UPDATED_RATIO}
	for trial := 0; trial < *numTrials; trial++ {
		InfoLogger.Printf("----- TRIAL: %09d -----", trial)

		floatDimension := dimensionDist.Rand()
		dimension := int(floatDimension)
		numTargetStates := int(floatDimension * targetStatesRatioDist.Rand())
		unitsUpdated := int(floatDimension * unitsUpdatedRatioDist.Rand())

		InfoLogger.Printf("Dimension: %v\n", dimension)
		InfoLogger.Printf("Num Target States: %v\n", numTargetStates)
		InfoLogger.Printf("Units Updated: %v\n", unitsUpdated)
		InfoLogger.Printf("Test States: %v\n", *numTestStates)

		network := hopfieldnetwork.NewHopfieldNetworkBuilder().
			SetNetworkDimension(dimension).
			SetNetworkDomain(DOMAIN).
			SetRandMatrixInit(false).
			SetNetworkLearningRule(hopfieldnetwork.DeltaLearningRule).
			SetEpochs(100).
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
		for _, result := range testResults {
			if result {
				numStable += 1
			}
		}

		InfoLogger.Printf("Stable Test States: %v\n", numStable)

		datawriter.Write([]string{strconv.Itoa(dimension), strconv.Itoa(numTargetStates), strconv.Itoa(unitsUpdated), strconv.Itoa(*numTestStates), strconv.Itoa(numStable)})
	}
}
