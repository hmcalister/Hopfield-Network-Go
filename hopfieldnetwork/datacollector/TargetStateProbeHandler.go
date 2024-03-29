package datacollector

import "github.com/xitongsys/parquet-go/writer"

// TargetStateIndex is the index of the target state being probed
// IsStable is a flag indicating if this target state is stable in the network
// State is the state vector
// EnergyProfile is the energy profile of the target state in the network
type TargetStateProbeData struct {
	TargetStateIndex int       `parquet:"name=TargetStateIndex, type=INT32"`
	IsStable         bool      `parquet:"name=IsStable, type=BOOLEAN"`
	State            []float64 `parquet:"name=State, type=DOUBLE, repetitiontype=REPEATED"`
	EnergyProfile    []float64 `parquet:"name=EnergyProfile, type=DOUBLE, repetitiontype=REPEATED"`
}

func NewTargetStateProbeHandler(dataFile string) *dataHandler {
	fileHandle, dataWriter := newParquetWriter(dataFile, new(TargetStateProbeData))
	return &dataHandler{
		eventID:     DataCollectionEvent_TargetStateProbe,
		dataWriter:  dataWriter,
		fileHandle:  fileHandle,
		handleEvent: handleTargetStateProbeEvent,
		cleanupFn:   defaultCleanupFn,
	}
}

func handleTargetStateProbeEvent(writer *writer.ParquetWriter, event interface{}) {
	result := event.(TargetStateProbeData)
	writer.Write(result)
}
