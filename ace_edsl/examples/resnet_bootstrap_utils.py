#!/usr/bin/env python3
"""Utilities for integrating generated primitive bootstrap into resnet runs."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import textwrap
from pathlib import Path


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _bootstrap_module():
    script_dir = _script_dir()
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import bootstrap_full

    return bootstrap_full


def _with_env(**updates):
    class _EnvCtx:
        def __enter__(self):
            self._old = {key: os.environ.get(key) for key in updates}
            for key, value in updates.items():
                os.environ[key] = str(value)

        def __exit__(self, exc_type, exc, tb):
            for key, value in self._old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    return _EnvCtx()


def generate_demo(args: argparse.Namespace) -> int:
    with _with_env(
        ACE_BOOTSTRAP_IMPL=args.impl,
        ACE_BOOTSTRAP_POLY_DEGREE=args.poly_degree,
        ACE_BOOTSTRAP_MUL_LEVEL=args.mul_level,
        ACE_BOOTSTRAP_FUNCTION_NAME_PREFIX=args.function_name_prefix,
        ACE_BOOTSTRAP_CONSTANT_NAME_PREFIX=args.const_prefix,
        ACE_BOOTSTRAP_PT_FROM_MSG_NAME=args.pt_from_msg_name,
        ACE_BOOTSTRAP_RAISE_LEVEL_NAME=args.raise_level_name,
        ACE_BOOTSTRAP_RUNTIME_RAISE_LEVEL=(
            "1" if args.runtime_raise_level else "0"
        ),
    ):
        bootstrap_full = _bootstrap_module()
        importlib.reload(bootstrap_full)
        ok = bootstrap_full.run_demo()
    return 0 if ok else 1


def emit_body(args: argparse.Namespace) -> int:
    src_path = Path(args.bootstrap_c)
    dst_path = Path(args.output)

    dst_path.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")
    return 0


def emit_shim(args: argparse.Namespace) -> int:
    dst_path = Path(args.output)
    source = textwrap.dedent(
        f"""\
        #include <stdlib.h>
        #include <stdio.h>
        #include <string.h>
        #include <time.h>
        #include <fcntl.h>
        #include <unistd.h>
        #include "ckks/cipher.h"
        #include "ckks/ciphertext.h"
        #include "common/rt_api.h"
        #include "fhe/core/rt_data_def.h"
        #include "fhe/core/rt_encode_api.h"

        #ifdef __cplusplus
        extern "C" {{
        #endif

        CKKS_PARAMS* Get_context_params(void);
        CIPHERTEXT {args.entry_name}(CIPHERTEXT p0, CIPHERTEXT p1);
        RT_DATA_INFO* {args.rtdata_name}(void);
        CKKS_PARAMS* {args.ctxparams_name}(void);
        PLAIN {args.pt_from_msg_name}(void* pt, uint32_t index, size_t len,
                                      uint32_t scale, uint32_t level);

        static unsigned long g_dsl_bts_call_counter = 0;
        static __thread uint32_t g_dsl_bts_target_level_after_bts = 0;

        CKKS_PARAMS* Get_extra_context_params(void) {{
          return {args.ctxparams_name}();
        }}

        static double dsl_bts_now_sec(void) {{
          struct timespec ts;
          clock_gettime(CLOCK_MONOTONIC, &ts);
          return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
        }}

        typedef struct {{
          int active;
          int phase;
          double last_sec;
        }} DSL_BTS_STAGE_PROBE;

        static __thread DSL_BTS_STAGE_PROBE g_dsl_bts_stage_probe = {{0, 0, 0.0}};

        static int dsl_bts_env_flag_enabled(const char* name) {{
          const char* flag = getenv(name);
          return flag != NULL && flag[0] != '\\0' && strcmp(flag, "0") != 0 &&
                 strcmp(flag, "false") != 0 && strcmp(flag, "off") != 0 &&
                 strcmp(flag, "no") != 0;
        }}

        static int dsl_bts_stage_probe_enabled(void) {{
          return dsl_bts_env_flag_enabled("ACE_BOOTSTRAP_STAGE_PROBE");
        }}

        static void dsl_bts_stage_probe_begin(void) {{
          if (!dsl_bts_stage_probe_enabled()) {{
            memset(&g_dsl_bts_stage_probe, 0, sizeof(g_dsl_bts_stage_probe));
            return;
          }}
          g_dsl_bts_stage_probe.active = 1;
          g_dsl_bts_stage_probe.phase = 0;
          g_dsl_bts_stage_probe.last_sec = dsl_bts_now_sec();
        }}

        static void dsl_bts_stage_probe_mark(const char* stage) {{
          if (!g_dsl_bts_stage_probe.active) {{
            return;
          }}
          double now = dsl_bts_now_sec();
          fprintf(stderr, "[{args.log_prefix}_stage] %s elapsed=%.3fs\\n",
                  stage, now - g_dsl_bts_stage_probe.last_sec);
          fflush(stderr);
          g_dsl_bts_stage_probe.last_sec = now;
        }}

        static void dsl_bts_stage_probe_finish(void) {{
          if (!g_dsl_bts_stage_probe.active) {{
            return;
          }}
          if (g_dsl_bts_stage_probe.phase == 4) {{
            dsl_bts_stage_probe_mark("slots_to_coeffs");
            g_dsl_bts_stage_probe.phase = 5;
          }}
          if (g_dsl_bts_stage_probe.phase == 5) {{
            dsl_bts_stage_probe_mark("post_scale");
          }}
          memset(&g_dsl_bts_stage_probe, 0, sizeof(g_dsl_bts_stage_probe));
        }}

        CIPHER dsl_bts_probe_Conjugate_ciph(CIPHER res, CIPHER ciph) {{
          if (g_dsl_bts_stage_probe.active &&
              g_dsl_bts_stage_probe.phase == 0) {{
            dsl_bts_stage_probe_mark("coeff_to_slots");
            g_dsl_bts_stage_probe.phase = 1;
          }}
          return Conjugate_ciph(res, ciph);
        }}

        CIPHER dsl_bts_probe_Mul_mono_ciph(CIPHER res, CIPHER ciph,
                                           uint32_t power) {{
          if (g_dsl_bts_stage_probe.active &&
              g_dsl_bts_stage_probe.phase == 1) {{
            CIPHER out = Mul_mono_ciph(res, ciph, power);
            dsl_bts_stage_probe_mark("split");
            g_dsl_bts_stage_probe.phase = 2;
            return out;
          }}
          if (g_dsl_bts_stage_probe.active &&
              g_dsl_bts_stage_probe.phase == 2) {{
            dsl_bts_stage_probe_mark("dual_evalmod");
            g_dsl_bts_stage_probe.phase = 3;
          }}
          return Mul_mono_ciph(res, ciph, power);
        }}

        void dsl_bts_probe_Init_ciph_same_scale(CIPHER res, CIPHER ciph1,
                                                CIPHER ciph2) {{
          if (g_dsl_bts_stage_probe.active &&
              g_dsl_bts_stage_probe.phase == 4 && ciph1 != NULL &&
              ciph1 == ciph2) {{
            dsl_bts_stage_probe_mark("slots_to_coeffs");
            g_dsl_bts_stage_probe.phase = 5;
          }}
          Init_ciph_same_scale(res, ciph1, ciph2);
        }}

        uint32_t {args.raise_level_name}(void) {{
          CKKS_PARAMS* ctx = Get_context_params();
          if (ctx == NULL) {{
            ctx = {args.ctxparams_name}();
          }}
          if (ctx == NULL) {{
            fprintf(stderr, "[{args.log_prefix}] missing bootstrap context params\\n");
            fflush(stderr);
            abort();
          }}
          uint32_t q_cnt = (uint32_t)(ctx->_mul_depth + 1);
          uint32_t target_level = g_dsl_bts_target_level_after_bts;
          uint32_t bts_depth = {args.bootstrap_depth};
          if (target_level == 0) {{
            return q_cnt;
          }}
          if (target_level > q_cnt - bts_depth) {{
            fprintf(stderr,
                    "[{args.log_prefix}] target level %u exceeds max %u for bootstrap depth %u\\n",
                    target_level, q_cnt - bts_depth, bts_depth);
            fflush(stderr);
            abort();
          }}
          return target_level + bts_depth;
        }}

        static void dsl_bts_dump_first_output(CIPHER ciph) {{
          uint32_t dump_len = Get_ciph_slots(ciph);
          if (dump_len > 8) {{
            dump_len = 8;
          }}
          Print_cipher_msg_with_imag(stderr, "{args.dump_label}", ciph, dump_len);
          fflush(stderr);
        }}

        static void dsl_bts_maybe_stop_after_first_dump(void) {{
          const char* flag = getenv("ACE_STOP_AFTER_FIRST_BTS");
          if (flag != NULL && flag[0] != '\\0' && strcmp(flag, "0") != 0) {{
            fprintf(stderr, "[{args.log_prefix}] stopping after first bootstrap dump\\n");
            fflush(stderr);
            _Exit(0);
          }}
        }}

        typedef struct {{
          int fd;
          struct DATA_FILE_HDR hdr;
          struct DATA_LUT_ENTRY* lut;
          int initialized;
        }} DSL_BTS_PT_FILE;

        static DSL_BTS_PT_FILE g_dsl_bts_pt = {{-1, {{0}}, NULL, 0}};
        static __thread char* g_dsl_bts_pt_tls_buf = NULL;
        static __thread uint64_t g_dsl_bts_pt_tls_size = 0;

        static void dsl_bts_pt_fini(void) {{
          if (!g_dsl_bts_pt.initialized) {{
            return;
          }}
          if (g_dsl_bts_pt.fd >= 0) {{
            close(g_dsl_bts_pt.fd);
          }}
          free(g_dsl_bts_pt.lut);
          g_dsl_bts_pt.fd = -1;
          g_dsl_bts_pt.lut = NULL;
          g_dsl_bts_pt.initialized = 0;
        }}

        static char* dsl_bts_pt_thread_buf(uint64_t min_size) {{
          if (g_dsl_bts_pt_tls_size >= min_size && g_dsl_bts_pt_tls_buf != NULL) {{
            return g_dsl_bts_pt_tls_buf;
          }}
          char* new_buf = (char*)realloc(g_dsl_bts_pt_tls_buf, min_size);
          if (new_buf == NULL) {{
            fprintf(stderr, "[{args.log_prefix}] failed to grow bootstrap plaintext scratch buffer\\n");
            fflush(stderr);
            abort();
          }}
          g_dsl_bts_pt_tls_buf = new_buf;
          g_dsl_bts_pt_tls_size = min_size;
          return g_dsl_bts_pt_tls_buf;
        }}

        static void dsl_bts_pt_init(void) {{
          if (g_dsl_bts_pt.initialized) {{
            return;
          }}
          RT_DATA_INFO* info = {args.rtdata_name}();
          if (info == NULL || info->_file_name == NULL || info->_file_name[0] == '\\0') {{
            fprintf(stderr, "[{args.log_prefix}] missing bootstrap rt data info\\n");
            fflush(stderr);
            abort();
          }}
          g_dsl_bts_pt.fd = open(info->_file_name, O_RDONLY);
          if (g_dsl_bts_pt.fd < 0) {{
            perror("[{args.log_prefix}] open bootstrap plaintext file");
            abort();
          }}
          ssize_t ret = pread(g_dsl_bts_pt.fd, &g_dsl_bts_pt.hdr,
                              sizeof(struct DATA_FILE_HDR), 0);
          if (ret != (ssize_t)sizeof(struct DATA_FILE_HDR)) {{
            fprintf(stderr, "[{args.log_prefix}] failed to read bootstrap plaintext header\\n");
            fflush(stderr);
            abort();
          }}
          uint64_t lut_size =
              sizeof(struct DATA_LUT_ENTRY) * g_dsl_bts_pt.hdr._ent_count;
          g_dsl_bts_pt.lut = (struct DATA_LUT_ENTRY*)malloc(lut_size);
          if (g_dsl_bts_pt.lut == NULL) {{
            fprintf(stderr, "[{args.log_prefix}] failed to allocate bootstrap LUT\\n");
            fflush(stderr);
            abort();
          }}
          ret = pread(g_dsl_bts_pt.fd, g_dsl_bts_pt.lut, lut_size,
                      g_dsl_bts_pt.hdr._lut_ofst);
          if (ret != (ssize_t)lut_size) {{
            fprintf(stderr, "[{args.log_prefix}] failed to read bootstrap LUT\\n");
            fflush(stderr);
            abort();
          }}
          g_dsl_bts_pt.initialized = 1;
          atexit(dsl_bts_pt_fini);
        }}

        PLAIN {args.pt_from_msg_name}(void* pt, uint32_t index, size_t len,
                                      uint32_t scale, uint32_t level) {{
          (void)pt;
          (void)len;
          (void)scale;
          (void)level;
          dsl_bts_pt_init();
          // The generated primitive bootstrap body immediately Copy_plain(...)
          // from each Pt_from_msg(...) result, so a thread-local scratch buffer
          // is sufficient and avoids keeping every bootstrap plaintext resident.
          if (index >= g_dsl_bts_pt.hdr._ent_count) {{
            fprintf(stderr, "[{args.log_prefix}] bootstrap plaintext index out of range: %u\\n", index);
            fflush(stderr);
            abort();
          }}
          struct DATA_LUT_ENTRY* lut = &g_dsl_bts_pt.lut[index];
          if (g_dsl_bts_stage_probe.active &&
              g_dsl_bts_stage_probe.phase == 3) {{
            dsl_bts_stage_probe_mark("recombine");
            g_dsl_bts_stage_probe.phase = 4;
          }}
          char* slot_ptr = dsl_bts_pt_thread_buf(lut->_size);
          ssize_t ret = pread(g_dsl_bts_pt.fd, slot_ptr, lut->_size, lut->_ent_ofst);
          if (ret != (ssize_t)lut->_size) {{
            fprintf(stderr, "[{args.log_prefix}] failed to read bootstrap plaintext entry %u\\n", index);
            fflush(stderr);
            abort();
          }}
          return (PLAIN)Cast_buffer_to_plain((struct PLAINTEXT_BUFFER*)slot_ptr);
        }}

        CIPHER {args.bootstrap_call_name}(CIPHER res, CIPHER ciph,
                                          uint32_t level_after_bts,
                                          uint32_t num_slots) {{
          unsigned long call_id = __sync_add_and_fetch(&g_dsl_bts_call_counter, 1);
          size_t in_level = Level(ciph);
          uint32_t in_sfdeg = Sc_degree(ciph);
          uint32_t in_slots = Get_ciph_slots(ciph);
          double t0 = dsl_bts_now_sec();
          fprintf(stderr,
                  "[{args.log_prefix}] begin call=%lu target_level=%u num_slots=%u "
                  "in_level=%zu in_sfdeg=%u in_slots=%u\\n",
                  call_id, level_after_bts, num_slots, in_level, in_sfdeg, in_slots);

          CIPHERTEXT in_copy;
          memset(&in_copy, 0, sizeof(in_copy));
          Copy_ciphertext(&in_copy, ciph);

          uint32_t prev_target_level = g_dsl_bts_target_level_after_bts;
          g_dsl_bts_target_level_after_bts = level_after_bts;
          dsl_bts_stage_probe_begin();
          CIPHERTEXT out = {args.entry_name}(in_copy, in_copy);
          dsl_bts_stage_probe_finish();
          g_dsl_bts_target_level_after_bts = prev_target_level;
          if (level_after_bts != 0) {{
            while (Level(&out) > level_after_bts) {{
              Modswitch_ciph(&out);
            }}
          }}
          double elapsed = dsl_bts_now_sec() - t0;
          fprintf(stderr,
                  "[{args.log_prefix}] end   call=%lu out_level=%zu out_sfdeg=%u out_slots=%u "
                  "elapsed=%.3fs\\n",
                  call_id, Level(&out), Sc_degree(&out), Get_ciph_slots(&out), elapsed);
          if (call_id == 1) {{
            dsl_bts_dump_first_output(&out);
            dsl_bts_maybe_stop_after_first_dump();
          }}

          Free_poly_data(Get_c0(&in_copy));
          Free_poly_data(Get_c1(&in_copy));

          if (res == ciph) {{
            Free_poly_data(Get_c0(res));
            Free_poly_data(Get_c1(res));
          }}

          *res = out;
          return res;
        }}

        #ifdef __cplusplus
        }}
        #endif
        """
    )
    dst_path.write_text(source, encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    generate = subparsers.add_parser("generate-demo")
    generate.add_argument("--impl", default="primitive")
    generate.add_argument("--poly-degree", type=int, required=True)
    generate.add_argument("--mul-level", type=int, required=True)
    generate.add_argument("--function-name-prefix", default="dsl_bts_")
    generate.add_argument("--const-prefix", default="dsl_bts")
    generate.add_argument("--pt-from-msg-name", default="dsl_bts_Pt_from_msg")
    generate.add_argument("--raise-level-name", default="dsl_bts_raise_level")
    generate.add_argument(
        "--runtime-raise-level",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    generate.set_defaults(func=generate_demo)

    emit = subparsers.add_parser("emit-body")
    emit.add_argument("--bootstrap-c", required=True)
    emit.add_argument("--output", required=True)
    emit.add_argument("--ctxparams-name", default="dsl_bts_Get_context_params")
    emit.add_argument("--entry-name", default="dsl_bts_bootstrap_full")
    emit.add_argument("--rtdata-name", default="dsl_bts_Get_rt_data_info")
    emit.add_argument("--pt-from-msg-name", default="dsl_bts_Pt_from_msg")
    emit.add_argument("--rotate-name", default="dsl_bts_Rotate")
    emit.add_argument("--relin-name", default="dsl_bts_Relinearize")
    emit.add_argument("--raise-level-name", default="dsl_bts_raise_level")
    emit.add_argument("--const-prefix", default="dsl_bts")
    emit.set_defaults(func=emit_body)

    shim = subparsers.add_parser("emit-shim")
    shim.add_argument("--output", required=True)
    shim.add_argument("--entry-name", default="dsl_bts_bootstrap_full")
    shim.add_argument("--ctxparams-name", default="dsl_bts_Get_context_params")
    shim.add_argument("--rtdata-name", default="dsl_bts_Get_rt_data_info")
    shim.add_argument("--pt-from-msg-name", default="dsl_bts_Pt_from_msg")
    shim.add_argument("--raise-level-name", default="dsl_bts_raise_level")
    shim.add_argument("--bootstrap-depth", type=int, default=15)
    shim.add_argument("--bootstrap-call-name", default="Eval_bootstrap_ciph_dsl")
    shim.add_argument("--log-prefix", default="dsl_bts")
    shim.add_argument("--dump-label", default="dsl_bts_round1")
    shim.set_defaults(func=emit_shim)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
