package datacollector

import "github.com/xitongsys/parquet-go/writer"

type RelaxationHistoryData struct {
	TrialIndex    int       `parquet:"name=TrialIndex, type=INT32"`
	StateIndex    int       `parquet:"name=StateIndex, type=INT32"`
	HistoryIndex  int       `parquet:"name=HistoryIndex, type=INT32"`
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
