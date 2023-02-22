package datacollector

type OnTrialEndData struct {
	TrialIndex                 int     `parquet:"name=TrialIndex, type=INT32"`
	NumberStableStates         int     `parquet:"name=NumberStableStates, type=INT32"`
	StableStatesMeanStepsTaken float64 `parquet:"name=StableStatesMeanStepsTaken, type=DOUBLE"`
}

type onTrialEndHandler struct {
	defaultDataHandler
}

func newOnTrialEndHandler(dataFile string) *onTrialEndHandler {
	return &onTrialEndHandler{
		defaultDataHandler: defaultDataHandler{
			eventID:    DataCollectionEvent_OnTrialEnd,
			dataWriter: newParquetWriter(dataFile, new(OnTrialEndData)),
		},
	}
}

func (collector *onTrialEndHandler) handleEvent(event interface{}) {
	result := event.(OnTrialEndData)
	collector.dataWriter.Write(result)
}
