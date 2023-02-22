package datacollector

import (
	"github.com/xitongsys/parquet-go-source/local"
	"github.com/xitongsys/parquet-go/parquet"
	"github.com/xitongsys/parquet-go/writer"
)

// Define the handler interface which all handlers must implement
//
// getEventID allows the DataCollector to determine if this handler responds to the given ID.
// In future this may change to a slice of ints so a handler can respond to many events
//
// handleEvent is the core functionality of a handler, where the event given to the
// data collector is cast to a struct and written to a file.
//
// writeStop is a function to stop writing and close the file.
type handler interface {
	getEventID() int
	handleEvent(event interface{})
	writeStop()
}

// The default data handler (embedded by most handler structs directly)
// implements getEventID and writeStop already, as well as dealing with
// the eventID and writer fields.
type defaultDataHandler struct {
	eventID    int
	dataWriter *writer.ParquetWriter
}

func (handler *defaultDataHandler) getEventID() int {
	return handler.eventID
}

func (handler *defaultDataHandler) writeStop() {
	handler.dataWriter.WriteStop()
}

// Create a new parquet writer to a given file path, using a given struct.
//
// This is a utility method to avoid the same boilerplate code over and over.
//
// See this example (https://github.com/xitongsys/parquet-go/blob/master/example/local_flat.go)
// for information on how to format the structs and use this method nicely.
//
// It may be wise to call `defer writer.WriteStop()` after calling this method!
//
// # Arguments
//
// * `dataFilePath`: The path to the data file required
//
// * `dataStruct`: A valid struct for writing in the parquet format. Should be called with
// new(struct) as argument.
//
// # Returns
//
// A ParquetWriter to the data file in question.
func newParquetWriter[T interface{}](dataFilePath string, dataStruct T) *writer.ParquetWriter {
	dataFileWriter, _ := local.NewLocalFileWriter(dataFilePath)
	parquetDataWriter, _ := writer.NewParquetWriter(dataFileWriter, dataStruct, 1)
	parquetDataWriter.RowGroupSize = 128 * 1024 * 1024 //128MB
	parquetDataWriter.PageSize = 8 * 1024              //8K
	parquetDataWriter.CompressionType = parquet.CompressionCodec_SNAPPY
	parquetDataWriter.Flush(true)

	return parquetDataWriter
}
