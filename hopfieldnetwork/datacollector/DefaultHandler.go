package datacollector

import (
	"github.com/xitongsys/parquet-go-source/local"
	"github.com/xitongsys/parquet-go/parquet"
	"github.com/xitongsys/parquet-go/writer"
)

// Define the handleEvent function to be a template across all handlers
type handleEventFn func(*writer.ParquetWriter, interface{})

// A dataHandler specifies what events it listens to, and what to do when that event occurs
type dataHandler struct {
	eventID     int
	dataWriter  *writer.ParquetWriter
	handleEvent handleEventFn
}

func (handler *dataHandler) getEventID() int {
	return handler.eventID
}

func (handler *dataHandler) writeStop() error {
	return handler.dataWriter.WriteStop()
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
	parquetDataWriter, _ := writer.NewParquetWriter(dataFileWriter, dataStruct, 4)
	parquetDataWriter.RowGroupSize = 128 * 1024 * 1024 //128MB
	parquetDataWriter.PageSize = 8 * 1024              //8K
	parquetDataWriter.CompressionType = parquet.CompressionCodec_SNAPPY
	parquetDataWriter.Flush(true)

	return parquetDataWriter
}
