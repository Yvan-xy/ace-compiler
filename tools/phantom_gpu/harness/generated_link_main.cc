#include "common/rt_api.h"

extern "C" {
int Get_input_count() { return 0; }

int Get_output_count() { return 0; }

DATA_SCHEME* Get_encode_scheme(int) { return nullptr; }

DATA_SCHEME* Get_decode_scheme(int) { return nullptr; }
}

int main() { return 0; }
