// Process CPU time includes every OpenMP worker. Mark only on the main thread.
// Buffer output until the invocation finishes so logging is outside each stage.
#include "profile_hooks.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <time.h>

typedef struct {
  const char* stage;
  double time;
  struct rusage usage;
  unsigned q, scale_degree;
} Point;
static Point points[32];
static unsigned count, sample_index;
static int active;

void ace_profile_mark(const char* stage) {
  if (!active) return;
  if (count >= 32) abort();
  Point* p = &points[count++];
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  p->time = t.tv_sec + t.tv_nsec * 1e-9;
  p->stage = stage;
  p->q = p->scale_degree = 0;
  getrusage(RUSAGE_SELF, &p->usage);
}
void ace_profile_state(unsigned q, unsigned scale_degree) {
  if (active && count) {
    points[count - 1].q = q;
    points[count - 1].scale_degree = scale_degree;
  }
}
void ace_profile_begin(unsigned sample) {
  count = 0;
  sample_index = sample;
  active = 1;
  ace_profile_mark("entry");
}
static double tv(struct timeval t) { return t.tv_sec + t.tv_usec * 1e-6; }
void ace_profile_end(void) {
  ace_profile_mark("end");
  active = 0;
  for (unsigned i = 1; i < count; ++i) {
    Point* a = &points[i - 1];
    Point* b = &points[i];
    double user = tv(b->usage.ru_utime) - tv(a->usage.ru_utime);
    double sys = tv(b->usage.ru_stime) - tv(a->usage.ru_stime);
    printf("CPU_BTS_PHASE={\"sample\":%u,\"stage\":\"%s\","
           "\"start\":%.9f,\"end\":%.9f,\"seconds\":%.9f,"
           "\"cpu_seconds\":%.9f,\"system_seconds\":%.9f,"
           "\"minor_faults\":%ld,\"major_faults\":%ld,\"input_q\":%u,\"input_scale_degree\":%u}\n",
           sample_index, a->stage, a->time, b->time, b->time - a->time,
           user + sys, sys, b->usage.ru_minflt - a->usage.ru_minflt,
           b->usage.ru_majflt - a->usage.ru_majflt, a->q, a->scale_degree);
  }
  fflush(stdout);
}
