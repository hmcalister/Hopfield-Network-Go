package datacollector

import "github.com/xitongsys/parquet-go/writer"

// StateIndex is the index of the state in the probe collection
// StepIndex is the index of the states steps towards stability
// State is the value of the state in this instance
// EnergyProfile is the energy profile of the state in this instance
type RelaxationHistoryData struct {
	StateIndex    int       `parquet:"name=StateIndex, type=INT32"`
	StepIndex     int       `parquet:"name=StepIndex, type=INT32"`
	State         []float64 `parquet:"name=State, type=DOUBLE, repetitiontype=REPEATED"`
	EnergyProfile []float64 `parquet:"name=EnergyProfile, type=DOUBLE, repetitiontype=REPEATED"`
}

func NewRelaxationHistoryData(dataFile string) *dataHandler {
	fileHandle, dataWriter := newParquetWriter(dataFile, new(RelaxationHistoryData))
	return &dataHandler{
		eventID:     DataCollectionEvent_RelaxationHistory,
		dataWriter:  dataWriter,
		fileHandle:  fileHandle,
		handleEvent: handleRelaxationHistoryEvent,
	}
}

func handleRelaxationHistoryEvent(writer *writer.ParquetWriter, event interface{}) {
	result := event.(RelaxationHistoryData)
	writer.Write(result)
}
