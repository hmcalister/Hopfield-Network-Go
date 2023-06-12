package datacollector

// Representation of the Hopfield network data
// NetworkDimension is the dimension of the network
// LearningRule is the network learning rule (as a string)
// Epochs is the number of epochs the network is trained for
// LearningNoiseMethod is the method of noise used during training (as a string)
// LearningNoiseScale is the scale of the noise applied to the network
// UnitsUpdated is the amount of units updated at each step
// ForceSymmetricWeightMatrix is a boolean flag indicating if the network is allowed to take asymmetric values
// Threads is the number of threads the network used to relax states
// TargetStates is the number of states used for learning
// ProbeStates is the number of states used for probing
type HopfieldNetworkSummaryData struct {
	NetworkDomain               string  `parquet:"name=NetworkDomain, type=BYTE_ARRAY"`
	NetworkDimension            int     `parquet:"name=NetworkDimension, type=INT32"`
	LearningRule                string  `parquet:"name=LearningRule, type=BYTE_ARRAY"`
	Epochs                      int     `parquet:"name=Epochs, type=INT32"`
	MaximumRelaxationIterations int     `parquet:"name=MaximumRelaxationIterations, type=INT32"`
	LearningRate                float64 `parquet:"name=LearningRate, type=DOUBLE"`
	LearningNoiseMethod         string  `parquet:"name=LearningNoiseMethod, type=BYTE_ARRAY"`
	LearningNoiseScale          float64 `parquet:"name=LearningNoiseScale, type=DOUBLE"`
	UnitsUpdated                int     `parquet:"name=UnitsUpdated, type=INT32"`
	ForceSymmetricWeightMatrix  bool    `parquet:"name=ForceSymmetricWeightMatrix, type=BOOLEAN"`
	Threads                     int     `parquet:"name=Threads, type=INT32"`
	TargetStates                int     `parquet:"name=TargetStates, type=INT32"`
	ProbeStates                 int     `parquet:"name=ProbeStates, type=INT32"`
}

// Write the HopfieldNetworkSummary struct to the specified data file, in a parquet format
func WriteHopfieldNetworkSummary(dataFile string, summaryData *HopfieldNetworkSummaryData) error {
	fileHandle, dataWriter := newParquetWriter(dataFile, new(HopfieldNetworkSummaryData))
	dataWriter.Write(summaryData)
	if err := dataWriter.WriteStop(); err != nil {
		return err
	}
	if err := fileHandle.Close(); err != nil {
		return err
	}
	return nil
}
