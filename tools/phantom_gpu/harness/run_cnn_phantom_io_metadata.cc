#include "common/rt_api.h"

// The handwritten Phantom CNN owns its input and output files.  The ACE
// runtime still links io_lib, whose generated-program ABI requires these four
// metadata callbacks even though this executable never calls its tensor I/O
// path.  Keep their sole definitions in this narrow stitching object.
extern "C" {

int Get_input_count() { return 0; }

int Get_output_count() { return 0; }

DATA_SCHEME *Get_encode_scheme(int) { return nullptr; }

DATA_SCHEME *Get_decode_scheme(int) { return nullptr; }

} // extern "C"
