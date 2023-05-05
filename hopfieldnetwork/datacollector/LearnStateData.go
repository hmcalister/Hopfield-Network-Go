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
// DistancesToAllLearned is an array of distances to all learned states.
type LearnStateData struct {
	Epoch            int       `parquet:"name=Epoch, type=INT32"`
	TargetStateIndex int       `parquet:"name=TargetStateIndex, type=INT32"`
	EnergyProfile    []float64 `parquet:"name=EnergyProfile, type=DOUBLE, repetitiontype=REPEATED"`
	Stable           bool      `parquet:"name=Stable, type=BOOLEAN"`
}

func NewLearnStateHandler(dataFile string) *dataHandler {
	fileHandle, dataWriter := newParquetWriter(dataFile, new(LearnStateData))
	return &dataHandler{
		eventID:     DataCollectionEvent_LearnState,
		dataWriter:  dataWriter,
		fileHandle:  fileHandle,
		handleEvent: handleLearnStateEvent,
	}
}

func handleLearnStateEvent(writer *writer.ParquetWriter, event interface{}) {
	result := event.(LearnStateData)
	writer.Write(result)
}
