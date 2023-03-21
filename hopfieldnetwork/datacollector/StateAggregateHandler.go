package datacollector

import "github.com/xitongsys/parquet-go/writer"

type StateAggregateData struct {
	NumTestStates              int     `parquet:"name=NumTestStates, type=INT32"`
	NumTargetStates            int     `parquet:"name=NumTargetStates, type=INT32"`
	NumStableStates            int     `parquet:"name=NumStableStates, type=INT32"`
	StableStatesMeanStepsTaken float64 `parquet:"name=StableStatesMeanStepsTaken, type=DOUBLE"`
}

func NewStateAggregateHandler(dataFile string) *dataHandler {
	fileHandle, dataWriter := newParquetWriter(dataFile, new(StateAggregateData))
	return &dataHandler{
		eventID:     DataCollectionEvent_StateAggregate,
		dataWriter:  dataWriter,
		fileHandle:  fileHandle,
		handleEvent: handleStateAggregateEvent,
	}
}

func handleStateAggregateEvent(writer *writer.ParquetWriter, event interface{}) {
	result := event.(StateAggregateData)
	writer.Write(result)
}
