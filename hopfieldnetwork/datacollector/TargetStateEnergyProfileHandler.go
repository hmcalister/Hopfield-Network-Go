package datacollector

import "github.com/xitongsys/parquet-go/writer"

type TargetStateEnergyProfileData struct {
	TargetStateIndex int       `parquet:"name=TargetStateIndex, type=INT32"`
	IsStable         bool      `parquet:"name=IsStable, type=BOOLEAN"`
	EnergyProfile    []float64 `parquet:"name=EnergyProfile, type=DOUBLE, repetitiontype=REPEATED"`
}

func NewTargetStateEnergyProfileHandler(dataFile string) *dataHandler {
	fileHandle, dataWriter := newParquetWriter(dataFile, new(TargetStateEnergyProfileData))
	return &dataHandler{
		eventID:     DataCollectionEvent_TargetStateEnergyProfile,
		dataWriter:  dataWriter,
		fileHandle:  fileHandle,
		handleEvent: handleTargetStateEnergyProfileEvent,
	}
}

func handleTargetStateEnergyProfileEvent(writer *writer.ParquetWriter, event interface{}) {
	result := event.(TargetStateEnergyProfileData)
	writer.Write(result)
}
