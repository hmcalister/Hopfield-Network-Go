package hopfieldutils

import (
	"github.com/xitongsys/parquet-go-source/local"
	"github.com/xitongsys/parquet-go/parquet"
	"github.com/xitongsys/parquet-go/writer"
	"golang.org/x/exp/rand"
)

// Shuffles the given list
func ShuffleList[T comparable](randomGenerator *rand.Rand, list []T) {
	randomGenerator.Shuffle(len(list), func(i int, j int) {
		list[i], list[j] = list[j], list[i]
	})
}

// Defines a very simple wrapper to assign an index to another type.
//
// This type could be a *mat.VecDense to index states, or it could be an entire struct!
//
// In general an IndexedWrapper is useful to count particular types of the generic,
// such as passing items to a goroutine and tracking order before and after
type IndexedWrapper[T any] struct {
	Index int
	Data  T
}

// Allows a slice to be chunked into smaller slices. Returns a slice of slices.
//
// Note the final chunk may be smaller than chunkSize if there is a remainder upon division.
//
// # Arguments
//
// * `slice`: The slice to chunk.
//
// * `chunkSize`: The number of items to fit into each chunk.
//
// # Returns
//
// A slice of slices, where each internal slice (expect possibly the last one) has
// a number of items equal to chunkSize from the original list
func ChunkSlice[T any](slice []T, chunkSize int) [][]T {
	var chunkedSlices [][]T

	if chunkSize <= 0 {
		panic("chunkSize must be a positive integer!")
	}

	for i := 0; i < len(slice); i += chunkSize {
		chunkEnd := i + chunkSize

		// Ensure we do not run off the end of the array
		if chunkEnd > len(slice) {
			chunkEnd = len(slice)
		}

		chunkedSlices = append(chunkedSlices, slice[i:chunkEnd])
	}

	return chunkedSlices
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
func ParquetWriter[T interface{}](dataFilePath string, dataStruct T) *writer.ParquetWriter {
	dataFileWriter, _ := local.NewLocalFileWriter(dataFilePath)
	parquetDataWriter, _ := writer.NewParquetWriter(dataFileWriter, dataStruct, 1)
	parquetDataWriter.RowGroupSize = 128 * 1024 * 1024 //128MB
	parquetDataWriter.PageSize = 8 * 1024              //8K
	parquetDataWriter.CompressionType = parquet.CompressionCodec_SNAPPY

	return parquetDataWriter
}
