package datacollector

import (
	"os"

	"github.com/xitongsys/parquet-go-source/local"
	"github.com/xitongsys/parquet-go/parquet"
	"github.com/xitongsys/parquet-go/source"
	"github.com/xitongsys/parquet-go/writer"
)

// Define the handleEvent function to be a template across all handlers
type handleEventFn func(*writer.ParquetWriter, interface{})

type cleanupFn func(*writer.ParquetWriter)

func defaultCleanupFn(*writer.ParquetWriter) {}

// A dataHandler specifies what events it listens to, and what to do when that event occurs
type dataHandler struct {
	eventID     int
	dataWriter  *writer.ParquetWriter
	fileHandle  source.ParquetFile
	handleEvent handleEventFn
	cleanupFn   cleanupFn
}

func (handler *dataHandler) getEventID() int {
	return handler.eventID
}

func (handler *dataHandler) writeStop() error {
	handler.cleanupFn(handler.dataWriter)

	if err := handler.dataWriter.WriteStop(); err != nil {
		return err
	}
	if err := handler.fileHandle.Close(); err != nil {
		return err
	}
	return nil
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
// dataFilePath string: The path to the data file required
//
// dataStruct (generic struct): A valid struct for writing in the parquet format.
// Should be called with `new(dataStruct)` as argument.
//
// # Returns
//
// A ParquetWriter to the data file in question.
func newParquetWriter[T interface{}](dataFilePath string, dataStruct T) (source.ParquetFile, *writer.ParquetWriter) {
	os.Remove(dataFilePath)
	dataFileWriter, _ := local.NewLocalFileWriter(dataFilePath)
	parquetDataWriter, _ := writer.NewParquetWriter(dataFileWriter, dataStruct, 4)
	parquetDataWriter.RowGroupSize = 128 * 1024 * 1024 //128MB
	parquetDataWriter.PageSize = 8 * 1024              //8K
	parquetDataWriter.CompressionType = parquet.CompressionCodec_SNAPPY
	parquetDataWriter.Flush(true)

	return dataFileWriter, parquetDataWriter
}
