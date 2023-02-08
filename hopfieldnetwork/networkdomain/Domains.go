// Definitions of the network domains for the Hopfield Network.
package networkdomain

// Defines the NetworkDomain type as an int32 to work as an Enum.
//
// Note this is mainly done to enforce self documenting code.
type NetworkDomain int32

// The possible Network Domains represented as an Enum.
const (
	// Default domain - a sentinel to allow errors if building is attempted.
	UnspecifiedDomain NetworkDomain = iota

	// States can only have values in the set {0,1}.
	BinaryDomain NetworkDomain = iota

	// States can only have values in the set {-1,1}.
	BipolarDomain NetworkDomain = iota

	// States can have any value from the Real Numbers.
	// TODO: Allow continuous but bounded domains
	ContinuousDomain NetworkDomain = iota
)
