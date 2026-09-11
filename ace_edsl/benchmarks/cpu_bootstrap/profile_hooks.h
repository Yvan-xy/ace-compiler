#pragma once
#ifdef __cplusplus
extern "C" {
#endif
void ace_profile_begin(unsigned sample);
void ace_profile_mark(const char* stage);
void ace_profile_state(unsigned q, unsigned scale_degree);
void ace_profile_end(void);
#ifdef __cplusplus
}
#endif
