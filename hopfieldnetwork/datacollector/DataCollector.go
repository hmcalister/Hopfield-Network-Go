package datacollector

//
// Adding to the data collector:
//
// To add to the data collector, first identify what data you want to collect.
// If the data is a direct measurement (e.g. a RelaxationResult), easy!
// If it is something more derived (e.g. the number of stable states per trial) then add a variable to the var block below.
//
// Add a callback function to the callback section, add a channel to the data collector, initialize the channel in the new method
// in the StartCollecting method have the new channel call the callback function...
//
// This needs to be refactored...
//

import (
	"hmcalister/hopfield/hopfieldutils"
)

// ------------------------------------------------------------------------------------------------
// DATA COLLECTION EVENT ENUM
// ------------------------------------------------------------------------------------------------

const (
	DataCollectionEvent_OnStateRelax = iota
	DataCollectionEvent_OnTrialEnd   = iota
)

// ------------------------------------------------------------------------------------------------
// DATA COLLECTOR STRUCT AND BASE METHODS
// ------------------------------------------------------------------------------------------------

type DataCollector struct {
	handlers     []handler
	EventChannel chan hopfieldutils.IndexedWrapper[interface{}]
}

func (collector *DataCollector) CollectData() {
	if len(collector.handlers) == 0 {
		for {
			<-collector.EventChannel
		}
	}

	for {
		event := <-collector.EventChannel

		for _, handler := range collector.handlers {
			if handler.getEventID() == event.Index {
				handler.handleEvent(event.Data)
			}
		}
	}
}

// Create a new data writer.
//
// Note that by default the data writer will collect nothing, not responding to any callbacks.
// To add a data collection event, call one of the Add* methods on the resulting DataCollector object.
// This will make that callback trigger a collection event.
func NewDataCollector() *DataCollector {
	return &DataCollector{
		handlers:     make([]handler, 0),
		EventChannel: make(chan hopfieldutils.IndexedWrapper[interface{}], 100),
	}
}

func (collector *DataCollector) WriteStop() {
	for _, handler := range collector.handlers {
		handler.writeStop()
	}
}

// ------------------------------------------------------------------------------------------------
// DATA HANDLER ADDITION METHODS
// ------------------------------------------------------------------------------------------------

// Add a state relaxed event handler.
func (collector *DataCollector) AddStateRelaxedHandler(stateRelaxedDataFile string) *DataCollector {
	collector.handlers = append(collector.handlers, newOnStateRelaxedCollector(stateRelaxedDataFile))
	return collector
}

// Add a trial end event handler.
func (collector *DataCollector) AddOnTrialEndHandler(stateRelaxedDataFile string) *DataCollector {
	collector.handlers = append(collector.handlers, newOnTrialEndHandler(stateRelaxedDataFile))
	return collector
}
