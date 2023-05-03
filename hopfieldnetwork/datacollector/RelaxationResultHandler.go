package datacollector

import (
	"github.com/xitongsys/parquet-go/writer"
)

// Representation of the result of a relaxation of a state.
//
// State is the state vector that has been relaxed.
// UnitEnergies is a vector representing the energies of each unit.
// Stable is a bool representing if the state was stable when relaxation finished.
// NumSteps is an int representing the number of steps taken when relaxation finished.
// DistancesToTargets is an array of distances to all Targets states.
type RelaxationResultData struct {
	StateIndex         int       `parquet:"name=StateIndex, type=INT32"`
	Stable             bool      `parquet:"name=Stable, type=BOOLEAN"`
	NumSteps           int       `parquet:"name=NumSteps, type=INT32"`
	FinalState         []float64 `parquet:"name=FinalState, type=DOUBLE, repetitiontype=REPEATED"`
	DistancesToTargets []float64 `parquet:"name=DistancesToTargets, type=DOUBLE, repetitiontype=REPEATED"`
	EnergyProfile      []float64 `parquet:"name=EnergyProfile, type=DOUBLE, repetitiontype=REPEATED"`
}

func NewRelaxationResultHandler(dataFile string) *dataHandler {
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
