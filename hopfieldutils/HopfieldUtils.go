package hopfieldutils

import (
	"math"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
)

// Shuffles the given list
func ShuffleList[T comparable](randomGenerator *rand.Rand, list []T) {
	randomGenerator.Shuffle(len(list), func(i int, j int) {
		list[i], list[j] = list[j], list[i]
	})
}

// Checks if a given element is in a slice
//
// # Arguments
//
// slice: The slice to check over
//
// elem: The element to check for
//
// # Returns
//
// True if the element is present, false otherwise
func IsInSlice[T comparable](slice []T, elem T) bool {
	for _, item := range slice {
		if item == elem {
			return true
		}
	}

	return false
}

// Finds the distances from a given vector to all vectors in a slice
//
// # Arguments
//
// slice: The slice to check over
//
// vector: The element to check the distance to
//
// norm: The norm function to use. 1 for Hamming Distance, 2 for Euclidean...
//
// # Return
//
// A []float64 representing the distances from the given vector to the slice vectors
func DistancesToVectorCollection(slice []*mat.VecDense, vector *mat.VecDense, norm float64) []float64 {
	tempVec := mat.NewVecDense(vector.Len(), nil)
	distances := make([]float64, len(slice))
	var d1 float64
	var d2 float64
	for index, item := range slice {
		tempVec.SubVec(item, vector)
		d1 = tempVec.Norm(norm)

		tempVec.ScaleVec(-1.0, item)
		tempVec.SubVec(tempVec, vector)
		d2 = tempVec.Norm(norm)

		if d1 < d2 {
			distances[index] = d1
		} else {
			distances[index] = d2
		}
	}
	return distances
}

// Finds the distance from a given vector to the closest vector in a slice
//
// # Arguments
//
// slice: The slice to check over
//
// vector: The element to check the distance to
//
// norm: The norm function to use. 1 for Hamming Distance, 2 for Euclidean...
//
// # Return
//
// A float64 representing the distance from the given vector to the closest vector in the slice
func DistanceToClosestVec(slice []*mat.VecDense, vector *mat.VecDense, norm float64) float64 {
	minDist := math.Inf(+1)
	allDistances := DistancesToVectorCollection(slice, vector, norm)
	for _, item := range allDistances {
		if item < minDist {
			minDist = item
		}
	}
	return minDist
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
// slice: The slice to chunk.
//
// chunkSize: The number of items to fit into each chunk.
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
