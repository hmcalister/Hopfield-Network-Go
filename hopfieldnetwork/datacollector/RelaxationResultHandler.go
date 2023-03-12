package datacollector

import (
	"github.com/xitongsys/parquet-go/writer"
)

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
type RelaxationResultData struct {
	TrialIndex         int       `parquet:"name=TrialIndex, type=INT32"`
	StateIndex         int       `parquet:"name=StateIndex, type=INT32"`
	Stable             bool      `parquet:"name=Stable, type=BOOLEAN"`
	NumSteps           int       `parquet:"name=NumSteps, type=INT32"`
	DistancesToLearned []float64 `parquet:"name=DistancesToLearned, type=DOUBLE, repetitiontype=REPEATED"`
	EnergyProfile      []float64 `parquet:"name=EnergyProfile, type=DOUBLE, repetitiontype=REPEATED"`
}

// Add a state relaxed event handler.
func (collector *DataCollector) AddRelaxationResultHandler(relaxationResult string) *DataCollector {
	collector.handlers = append(collector.handlers, newRelaxationResultHandler(relaxationResult))
	return collector
}

func newRelaxationResultHandler(dataFile string) *dataHandler {
	fileHandle, dataWriter := newParquetWriter(dataFile, new(RelaxationResultData))
	return &dataHandler{
		eventID:     DataCollectionEvent_RelaxationResult,
		dataWriter:  dataWriter,
		fileHandle:  fileHandle,
		handleEvent: handleRelaxationResultEvent,
	}
}

func handleRelaxationResultEvent(writer *writer.ParquetWriter, event interface{}) {
	result := event.(RelaxationResultData)
	writer.Write(result)
}
