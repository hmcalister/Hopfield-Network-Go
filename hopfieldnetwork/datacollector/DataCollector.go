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
	DataCollectionEvent_RelaxationResult  = iota
	DataCollectionEvent_RelaxationHistory = iota
	DataCollectionEvent_StateAggregate    = iota
)

// ------------------------------------------------------------------------------------------------
// DATA COLLECTOR STRUCT AND BASE METHODS
// ------------------------------------------------------------------------------------------------

type DataCollector struct {
	handlers     []*dataHandler
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
				handler.handleEvent(handler.dataWriter, event.Data)
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
		handlers:     make([]*dataHandler, 0),
		EventChannel: make(chan hopfieldutils.IndexedWrapper[interface{}]),
	}
}

// Add a state relaxed event handler.
func (collector *DataCollector) AddHandler(dataHandler *dataHandler) *DataCollector {
	collector.handlers = append(collector.handlers, dataHandler)
	return collector
}

func (collector *DataCollector) WriteStop() error {
	for _, handler := range collector.handlers {
		if err := handler.writeStop(); err != nil {
			return err
		}
	}
	return nil
}
