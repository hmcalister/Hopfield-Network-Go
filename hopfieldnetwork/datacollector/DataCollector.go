package datacollector

import (
	"hmcalister/hopfield/hopfieldutils"
)

// ------------------------------------------------------------------------------------------------
// DATA COLLECTION EVENT ENUM
// ------------------------------------------------------------------------------------------------

const (
	DataCollectionEvent_RelaxationResult  = iota
	DataCollectionEvent_RelaxationHistory = iota
	DataCollectionEvent_TargetStateProbe  = iota
)

// ------------------------------------------------------------------------------------------------
// DATA COLLECTOR STRUCT AND BASE METHODS
// ------------------------------------------------------------------------------------------------

type DataCollector struct {
	handlers     []*dataHandler
	EventChannel chan hopfieldutils.IndexedWrapper[interface{}]
}

// Start data collection by spinning up a goroutine that forever looks at EventChannel,
// taking any incoming events and sending them to the listening handlers.
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
//
// # Arguments
//
// dataHandler *dataHandler: The handler to add to the collector
//
// # Returns
//
// A pointer to the DataCollector, to allow for chaining of AddHandler calls
func (collector *DataCollector) AddHandler(dataHandler *dataHandler) *DataCollector {
	collector.handlers = append(collector.handlers, dataHandler)
	return collector
}

// Call WriteStop on all parquet writers in the handlers. This means data should be written nicely to disk.
//
// Consider calling `defer collector.WriteStop()`
func (collector *DataCollector) WriteStop() error {
	for _, handler := range collector.handlers {
		if err := handler.writeStop(); err != nil {
			return err
		}
	}
	return nil
}
