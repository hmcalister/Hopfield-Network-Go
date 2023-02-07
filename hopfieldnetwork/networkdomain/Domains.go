package networkdomain

type NetworkDomain int32

const (
	// Default domain - a sentinel to allow errors if building is attempted
	UnspecifiedDomain NetworkDomain = iota

	// States can only have values in the set {0,1}
	BinaryDomain NetworkDomain = iota

	// States can only have values in the set {-1,1}
	BipolarDomain NetworkDomain = iota

	// States can have any value from the Real Numbers
	// TODO: Allow continuous but bounded domains
	ContinuousDomain NetworkDomain = iota
)
