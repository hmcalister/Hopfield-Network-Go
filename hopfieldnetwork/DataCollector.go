package hopfieldnetwork

import (
	"hmcalister/hopfield/hopfieldutils"

	"github.com/xitongsys/parquet-go-source/local"
	"github.com/xitongsys/parquet-go/parquet"
	"github.com/xitongsys/parquet-go/writer"
)

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

// ------------------------------------------------------------------------------------------------
// DATA COLLECTION TIMING ENUM
// ------------------------------------------------------------------------------------------------

const (
	dataCollector_OnStateRelaxed       = iota
	dataCollector_OnStableStateRelaxed = iota
	dataCollector_OnTrialEnd           = iota
)

// ------------------------------------------------------------------------------------------------
// DATA COLLECTION VARIABLES
// ------------------------------------------------------------------------------------------------

var (
	stableStateRelaxedCounter   int = 0
	unstableStateRelaxedCounter int = 0
	trialCounter                int = 0

	stableStateTotalSteps int = 0
)

// ------------------------------------------------------------------------------------------------
// DATA COLLECTOR STRUCT AND BASE METHODS
// ------------------------------------------------------------------------------------------------

type DataCollector struct {
	onStableStateRelaxedDataWriter   *writer.ParquetWriter
	onUnstableStateRelaxedDataWriter *writer.ParquetWriter
	onTrialEndDataWriter             *writer.ParquetWriter
	dataCollectionTimings            []int
}

// Create a new data writer.
//
// Note that by default the data writer will collect nothing, not responding to any callbacks.
// To add a data collection event, call one of the Add* methods on the resulting DataCollector object.
// This will make that callback trigger a collection event.
func NewDataCollector() *DataCollector {
	return &DataCollector{}
}

func (collector *DataCollector) WriteStop() {
	if collector.onUnstableStateRelaxedDataWriter != nil {
		collector.onUnstableStateRelaxedDataWriter.WriteStop()
	}
	if collector.onStableStateRelaxedDataWriter != nil {
		collector.onStableStateRelaxedDataWriter.WriteStop()
	}
}

// ------------------------------------------------------------------------------------------------
// ON STABLE STATE RELAXED
// ------------------------------------------------------------------------------------------------

type dataEntryOnStableStateRelaxed struct {
	TrialIndex         int       `parquet:"name=TrialIndex, type=INT32"`
	StateIndex         int       `parquet:"name=StateIndex, type=INT32"`
	StateEnergyVector  []float64 `parquet:"name=StateEnergyVector, type=DOUBLE, repetitiontype=REPEATED"`
	DistancesToLearned []float64 `parquet:"name=DistancesToLearned, type=DOUBLE, repetitiontype=REPEATED"`
}

func (collector *DataCollector) AddOnStableStateRelaxed(onStableStateDataFile string) *DataCollector {
	if hopfieldutils.IsInSlice(collector.dataCollectionTimings, dataCollector_OnStableStateRelaxed) {
		return collector
	}
	collector.dataCollectionTimings = append(collector.dataCollectionTimings, dataCollector_OnStableStateRelaxed)
	collector.onStableStateRelaxedDataWriter = newParquetWriter(onStableStateDataFile, new(dataEntryOnStableStateRelaxed))

	return collector
}

func (collector *DataCollector) CallbackStableStateRelaxed(relaxationResult *RelaxationResult) {
	defer func() { stableStateRelaxedCounter++ }()
	if !hopfieldutils.IsInSlice(collector.dataCollectionTimings, dataCollector_OnStableStateRelaxed) {
		return
	}

	stableStateTotalSteps += relaxationResult.NumSteps
	collector.onStableStateRelaxedDataWriter.Write(dataEntryOnStableStateRelaxed{
		TrialIndex:         trialCounter,
		StateIndex:         stableStateRelaxedCounter,
		StateEnergyVector:  relaxationResult.UnitEnergies,
		DistancesToLearned: relaxationResult.DistancesToLearned,
	})
}

// ------------------------------------------------------------------------------------------------
// ON UNSTABLE STATE RELAXED
// ------------------------------------------------------------------------------------------------

type dataEntryOnUnstableStateRelaxed struct {
	TrialIndex         int       `parquet:"name=TrialIndex, type=INT32"`
	StateIndex         int       `parquet:"name=StateIndex, type=INT32"`
	StateEnergyVector  []float64 `parquet:"name=StateEnergyVector, type=DOUBLE, repetitiontype=REPEATED"`
	DistancesToLearned []float64 `parquet:"name=DistancesToLearned, type=DOUBLE, repetitiontype=REPEATED"`
}

func (collector *DataCollector) AddOnUnstableStateRelaxed(onUnstableStateDataFile string) *DataCollector {
	if hopfieldutils.IsInSlice(collector.dataCollectionTimings, dataCollector_OnStateRelaxed) {
		return collector
	}
	collector.dataCollectionTimings = append(collector.dataCollectionTimings, dataCollector_OnStateRelaxed)
	collector.onUnstableStateRelaxedDataWriter = newParquetWriter(onUnstableStateDataFile, new(dataEntryOnUnstableStateRelaxed))

	return collector
}

func (collector *DataCollector) CallbackUnstableStateRelaxed(relaxationResult *RelaxationResult) {
	defer func() { unstableStateRelaxedCounter++ }()
	if !hopfieldutils.IsInSlice(collector.dataCollectionTimings, dataCollector_OnStateRelaxed) {
		return
	}

	collector.onUnstableStateRelaxedDataWriter.Write(dataEntryOnUnstableStateRelaxed{
		TrialIndex:         trialCounter,
		StateIndex:         unstableStateRelaxedCounter,
		StateEnergyVector:  relaxationResult.UnitEnergies,
		DistancesToLearned: relaxationResult.DistancesToLearned,
	})
}

// ------------------------------------------------------------------------------------------------
// ON TRIAL END
// ------------------------------------------------------------------------------------------------

type dataEntryOnTrialEnd struct {
	TrialIndex                 int     `parquet:"name=TrialIndex, type=INT32"`
	NumberStableStates         int     `parquet:"name=NumberStableStates, type=INT32"`
	StableStatesMeanStepsTaken float64 `parquet:"name=StableStatesMeanStepsTaken, type=DOUBLE"`
}

func (collector *DataCollector) AddOnTrialEnd(onTrialEndDataFile string) *DataCollector {
	if hopfieldutils.IsInSlice(collector.dataCollectionTimings, dataCollector_OnTrialEnd) {
		return collector
	}
	collector.dataCollectionTimings = append(collector.dataCollectionTimings, dataCollector_OnTrialEnd)
	collector.onTrialEndDataWriter = newParquetWriter(onTrialEndDataFile, new(dataEntryOnTrialEnd))

	return collector
}

func (collector *DataCollector) CallbackTrialEnd() {
	defer func() {
		trialCounter++
		unstableStateRelaxedCounter = 0
		stableStateRelaxedCounter = 0
		stableStateTotalSteps = 0
	}()

	if !hopfieldutils.IsInSlice(collector.dataCollectionTimings, dataCollector_OnTrialEnd) {
		return
	}

	collector.onTrialEndDataWriter.Write(dataEntryOnTrialEnd{
		TrialIndex:                 trialCounter,
		NumberStableStates:         stableStateRelaxedCounter,
		StableStatesMeanStepsTaken: float64(stableStateTotalSteps) / float64(stableStateRelaxedCounter),
	})
}
