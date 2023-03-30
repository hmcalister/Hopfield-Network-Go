package datacollector

import "github.com/xitongsys/parquet-go/writer"

type TargetStateProbeData struct {
	TargetStateIndex int       `parquet:"name=TargetStateIndex, type=INT32"`
	IsStable         bool      `parquet:"name=IsStable, type=BOOLEAN"`
	EnergyProfile    []float64 `parquet:"name=EnergyProfile, type=DOUBLE, repetitiontype=REPEATED"`
}

func NewTargetStateProbeHandler(dataFile string) *dataHandler {
	fileHandle, dataWriter := newParquetWriter(dataFile, new(TargetStateProbeData))
	return &dataHandler{
		eventID:     DataCollectionEvent_TargetStateProbe,
		dataWriter:  dataWriter,
		fileHandle:  fileHandle,
		handleEvent: handleTargetStateProbeEvent,
	}
}

func handleTargetStateProbeEvent(writer *writer.ParquetWriter, event interface{}) {
	result := event.(TargetStateProbeData)
	writer.Write(result)
}
