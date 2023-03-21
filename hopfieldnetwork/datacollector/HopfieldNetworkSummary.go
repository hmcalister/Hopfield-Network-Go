package datacollector

// Representation of the result of a relaxation of a state.
//
// State is the state vector that has been relaxed.
//
// UnitEnergies is a vector representing the energies of each unit.
//
// Stable is a bool representing if the state was stable when relaxation finished.
//
// NumSteps is an int representing the number of steps taken when relaxation finished.
//
// DistancesToAllLearned is an array of distances to all learned states.
type HopfieldNetworkSummaryData struct {
	NetworkDimension   int     `parquet:"name=NetworkDimension, type=INT32"`
	LearningRule       string  `parquet:"name=LearningRule, type=BYTE_ARRAY"`
	Epochs             int     `parquet:"name=Epochs, type=INT32"`
	LearningNoiseRatio float64 `parquet:"name=LearningNoiseRatio, type=DOUBLE"`
	UnitsUpdated       int     `parquet:"name=UnitsUpdated, type=INT32"`
	Trials             int     `parquet:"name=Trials, type=INT32"`
	Threads            int     `parquet:"name=Threads, type=INT32"`
	TargetStates       int     `parquet:"name=TargetStates, type=INT32"`
	TestStates         int     `parquet:"name=TestStates, type=INT32"`
}

func WriteHopfieldNetworkSummary(dataFile string, summaryData *HopfieldNetworkSummaryData) error {
	fileHandle, dataWriter := newParquetWriter(dataFile, new(HopfieldNetworkSummaryData))
	dataWriter.Write(summaryData)
	if err := dataWriter.WriteStop(); err != nil {
		return err
	}
	if err := fileHandle.Close(); err != nil {
		return err
	}
	return nil
}
