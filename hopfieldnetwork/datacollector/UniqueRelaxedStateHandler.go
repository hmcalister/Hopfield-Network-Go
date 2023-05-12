package datacollector

import (
	"fmt"

	"github.com/xitongsys/parquet-go/writer"
)

var (
	uniqueRelaxedStatesArray = []*UniqueRelaxedStateData{}
	uniqueRelaxedStatesMap   = map[string]*UniqueRelaxedStateData{}
)

// Representation of the result of a relaxation of a state.
//
// State is the state vector that has been relaxed.
// UnitEnergies is a vector representing the energies of each unit.
// Stable is a bool representing if the state was stable when relaxation finished.
// NumSteps is an int representing the number of steps taken when relaxation finished.
// DistancesToTargets is an array of distances to all Targets states.
type UniqueRelaxedStateData struct {
	StateIndex         int       `parquet:"name=StateIndex, type=INT32"`
	Stable             bool      `parquet:"name=Stable, type=BOOLEAN"`
	NumSteps           int       `parquet:"name=NumSteps, type=INT32"`
	FinalState         []float64 `parquet:"name=FinalState, type=DOUBLE, repetitiontype=REPEATED"`
	DistancesToTargets []float64 `parquet:"name=DistancesToTargets, type=DOUBLE, repetitiontype=REPEATED"`
	EnergyProfile      []float64 `parquet:"name=EnergyProfile, type=DOUBLE, repetitiontype=REPEATED"`
	Hits               int       `parquet:"name=Hits, type=INT32"`
}

func NewUniqueRelaxedStateHandler(dataFile string) *dataHandler {
	fileHandle, dataWriter := newParquetWriter(dataFile, new(UniqueRelaxedStateData))
	return &dataHandler{
		eventID:     DataCollectionEvent_RelaxationResult,
		dataWriter:  dataWriter,
		fileHandle:  fileHandle,
		handleEvent: handleUniqueRelaxedState,
		cleanupFn:   cleanupUniqueRelaxedState,
	}
}

func handleUniqueRelaxedState(writer *writer.ParquetWriter, event interface{}) {
	// Note result is coming from relaxation result, so we have to cast to that...
	relaxationResult := event.(RelaxationResultData)
	energyHash := fmt.Sprint(relaxationResult.EnergyProfile)

	// See if state has been seen before
	val, ok := uniqueRelaxedStatesMap[energyHash]
	if !ok {
		result := UniqueRelaxedStateData{
			StateIndex:         relaxationResult.StateIndex,
			Stable:             relaxationResult.Stable,
			NumSteps:           relaxationResult.NumSteps,
			FinalState:         relaxationResult.FinalState,
			DistancesToTargets: relaxationResult.DistancesToTargets,
			EnergyProfile:      relaxationResult.EnergyProfile,
			Hits:               1,
		}

		uniqueRelaxedStatesArray = append(uniqueRelaxedStatesArray, &result)
		uniqueRelaxedStatesMap[energyHash] = &result

		return
	} else {
		// The state HAS been seen before, so we simply increment hits
		val.Hits += 1
	}
}

func cleanupUniqueRelaxedState(writer *writer.ParquetWriter) {
	// Actually write all the structs we've stored
	for _, data := range uniqueRelaxedStatesArray {
		writer.Write(data)
	}
}
