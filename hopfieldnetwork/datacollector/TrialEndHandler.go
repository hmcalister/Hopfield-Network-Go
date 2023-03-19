package datacollector

import "github.com/xitongsys/parquet-go/writer"

type TrialEndData struct {
	TrialIndex                 int     `parquet:"name=TrialIndex, type=INT32"`
	NumTestStates              int     `parquet:"name=NumTestStates, type=INT32"`
	NumLearnedStates           int     `parquet:"name=NumTargetStates, type=INT32"`
	NumStableStates            int     `parquet:"name=NumStableStates, type=INT32"`
	StableStatesMeanStepsTaken float64 `parquet:"name=StableStatesMeanStepsTaken, type=DOUBLE"`
}

func NewTrialEndHandler(dataFile string) *dataHandler {
	fileHandle, dataWriter := newParquetWriter(dataFile, new(TrialEndData))
	return &dataHandler{
		eventID:     DataCollectionEvent_TrialEnd,
		dataWriter:  dataWriter,
		fileHandle:  fileHandle,
		handleEvent: handleTrialEndEvent,
	}
}

func handleTrialEndEvent(writer *writer.ParquetWriter, event interface{}) {
	result := event.(TrialEndData)
	writer.Write(result)
}
