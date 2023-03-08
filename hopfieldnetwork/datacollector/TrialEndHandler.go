package datacollector

import "github.com/xitongsys/parquet-go/writer"

type OnTrialEndData struct {
	TrialIndex                 int     `parquet:"name=TrialIndex, type=INT32"`
	NumTargetStates            int     `parquet:"name=NumTargetStates, type=INT32"`
	NumberStableStates         int     `parquet:"name=NumberStableStates, type=INT32"`
	StableStatesMeanStepsTaken float64 `parquet:"name=StableStatesMeanStepsTaken, type=DOUBLE"`
}

// Add a trial end event handler.
func (collector *DataCollector) AddOnTrialEndHandler(stateRelaxedDataFile string) *DataCollector {
	collector.handlers = append(collector.handlers, newOnTrialEndHandler(stateRelaxedDataFile))
	return collector
}

func newOnTrialEndHandler(dataFile string) *dataHandler {
	fileHandle, dataWriter := newParquetWriter(dataFile, new(OnTrialEndData))
	return &dataHandler{
		eventID:     DataCollectionEvent_OnTrialEnd,
		dataWriter:  dataWriter,
		fileHandle:  fileHandle,
		handleEvent: handleTrialEndEvent,
	}
}

func handleTrialEndEvent(writer *writer.ParquetWriter, event interface{}) {
	result := event.(OnTrialEndData)
	writer.Write(result)
}
