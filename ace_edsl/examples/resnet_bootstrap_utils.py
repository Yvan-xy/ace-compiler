#!/usr/bin/env python3
"""Utilities for integrating generated primitive bootstrap into resnet runs."""

from __future__ import annotations

import argparse
import importlib
import os
import re
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
    ):
        bootstrap_full = _bootstrap_module()
        importlib.reload(bootstrap_full)
        ok = bootstrap_full.run_demo()
    return 0 if ok else 1


def emit_body(args: argparse.Namespace) -> int:
    src_path = Path(args.bootstrap_c)
    dst_path = Path(args.output)

    out_lines: list[str] = []

    with src_path.open("r", encoding="utf-8") as source:
        for line in source:
            line = re.sub(r"\bGet_context_params\b", args.ctxparams_name, line)
            line = re.sub(r"\bbootstrap_full\b", args.entry_name, line)
            line = re.sub(r"\bGet_rt_data_info\b", args.rtdata_name, line)
            line = re.sub(r"\bPt_from_msg\b", args.pt_from_msg_name, line)
            line = re.sub(r"\bRotate\b", args.rotate_name, line)
            line = re.sub(r"\bRelinearize\b", args.relin_name, line)
            line = re.sub(r"\b(_cst_\d+)\b", rf"{args.const_prefix}\1", line)
            out_lines.append(line)

    body = "".join(out_lines)
    pt_decl = (
        f'void* {args.pt_from_msg_name}(void* pt, uint32_t index, size_t len, '
        f'uint32_t scale, uint32_t level);\n'
    )
    if pt_decl not in body:
        include_line = '#include "rt_ant/rt_ant.h"\n'
        body = body.replace(include_line, include_line + "\n" + pt_decl, 1)

    raise_decl = f"uint32_t {args.raise_level_name}(void);\n"
    if raise_decl not in body:
        include_line = '#include "rt_ant/rt_ant.h"\n'
        body = body.replace(include_line, include_line + raise_decl, 1)

    body = re.sub(
        r"Raise_mod\(\s*(&[^,]+)\s*,\s*(&[^,]+)\s*,\s*\d+\s*\);",
        rf"Raise_mod(\1, \2, {args.raise_level_name}());",
        body,
    )

    raw_stage_probe = os.environ.get("ACE_BOOTSTRAP_STAGE_PROBE", "").strip().lower()
    if raw_stage_probe and raw_stage_probe not in ("0", "false", "off", "no"):
        body = _inject_stage_probes(body)

    dst_path.write_text(body, encoding="utf-8")
    return 0


def _inject_stage_probes(body: str) -> str:
    support = textwrap.dedent(
        """\
        #include <time.h>
        #include <stdio.h>
        #include <stdlib.h>

        static int dsl_bts_stage_probe_enabled(void) {
          static int initialized = 0;
          static int enabled = 0;
          if (!initialized) {
            const char* flag = getenv("ACE_BOOTSTRAP_STAGE_PROBE");
            enabled = (flag != NULL && flag[0] != '\\0' && strcmp(flag, "0") != 0);
            initialized = 1;
          }
          return enabled;
        }

        static double dsl_bts_stage_now_sec(void) {
          struct timespec ts;
          clock_gettime(CLOCK_MONOTONIC, &ts);
          return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
        }

        static void dsl_bts_stage_mark(const char* stage, double* last) {
          if (!dsl_bts_stage_probe_enabled()) {
            return;
          }
          double now = dsl_bts_stage_now_sec();
          fprintf(stderr, "[dsl_bts_stage] %s elapsed=%.3fs\\n", stage, now - *last);
          fflush(stderr);
          *last = now;
        }
        """
    )

    include_anchor = 'void* dsl_bts_Pt_from_msg'
    if support not in body and include_anchor in body:
        body = body.replace(include_anchor, support + "\n" + include_anchor, 1)

    fn_anchor = "CIPHERTEXT dsl_bootstrap_full(CIPHERTEXT p0_0, CIPHERTEXT p1_1) {\n"
    if fn_anchor not in body:
        raise RuntimeError("stage probe injection failed: bootstrap entry anchor missing")
    body = body.replace(
        fn_anchor,
        fn_anchor +
        "  double _dsl_bts_stage_last = 0.0;\n"
        "  if (dsl_bts_stage_probe_enabled()) {\n"
        "    _dsl_bts_stage_last = dsl_bts_stage_now_sec();\n"
        "  }\n",
        1,
    )

    def insert_before_once(text: str, needle: str, snippet: str) -> str:
        idx = text.find(needle)
        if idx < 0:
            raise RuntimeError(f"stage probe injection failed: missing anchor {needle!r}")
        return text[:idx] + snippet + text[idx:]

    # Step 1 done: first full-packed CoeffToSlot result is ready before conjugate split.
    body = insert_before_once(
        body,
        "  Conjugate_ciph(",
        '  dsl_bts_stage_mark("coeff_to_slots", &_dsl_bts_stage_last);\n',
    )

    mul_mono_matches = list(re.finditer(r"^  Mul_mono_ciph\(", body, flags=re.M))
    if len(mul_mono_matches) < 2:
        raise RuntimeError("stage probe injection failed: expected two Mul_mono_ciph anchors")

    first_mul = mul_mono_matches[0].start()
    second_mul = mul_mono_matches[1].start()
    body = (
        body[:first_mul]
        + '  dsl_bts_stage_mark("split", &_dsl_bts_stage_last);\n'
        + body[first_mul:]
    )
    second_mul += len('  dsl_bts_stage_mark("split", &_dsl_bts_stage_last);\n')
    body = (
        body[:second_mul]
        + '  dsl_bts_stage_mark("dual_evalmod", &_dsl_bts_stage_last);\n'
        + body[second_mul:]
    )

    rotate_after_recombine = re.search(r"^  _preg_\d+ = dsl_bts_Rotate\(", body[second_mul:], flags=re.M)
    if rotate_after_recombine is None:
        raise RuntimeError("stage probe injection failed: missing SlotToCoeff rotate anchor")
    rotate_pos = second_mul + rotate_after_recombine.start()
    body = (
        body[:rotate_pos]
        + '  dsl_bts_stage_mark("recombine", &_dsl_bts_stage_last);\n'
        + body[rotate_pos:]
    )

    post_scale_pos = -1
    scan_pos = rotate_pos
    while True:
        line_end = body.find("\n", scan_pos)
        if line_end < 0:
            line_end = len(body)
        line = body[scan_pos:line_end]
        if line.startswith("  Init_ciph_same_scale("):
            args = line[len("  Init_ciph_same_scale("):-2]
            parts = [part.strip() for part in args.split(",")]
            if len(parts) == 3 and parts[1] == parts[2]:
                post_scale_pos = scan_pos
                break
        if line_end >= len(body):
            break
        scan_pos = line_end + 1
    if post_scale_pos < 0:
        raise RuntimeError("stage probe injection failed: missing post-scale doubling anchor")
    body = (
        body[:post_scale_pos]
        + '  dsl_bts_stage_mark("slots_to_coeffs", &_dsl_bts_stage_last);\n'
        + body[post_scale_pos:]
    )

    ret_pos = body.rfind("  return __ret_tmp_")
    if ret_pos < 0:
        raise RuntimeError("stage probe injection failed: missing bootstrap return anchor")
    body = (
        body[:ret_pos]
        + '  dsl_bts_stage_mark("post_scale", &_dsl_bts_stage_last);\n'
        + body[ret_pos:]
    )
    return body


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

        static double dsl_bts_now_sec(void) {{
          struct timespec ts;
          clock_gettime(CLOCK_MONOTONIC, &ts);
          return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
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
          CIPHERTEXT out = {args.entry_name}(in_copy, in_copy);
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
    generate.set_defaults(func=generate_demo)

    emit = subparsers.add_parser("emit-body")
    emit.add_argument("--bootstrap-c", required=True)
    emit.add_argument("--output", required=True)
    emit.add_argument("--ctxparams-name", default="Get_extra_context_params")
    emit.add_argument("--entry-name", default="dsl_bootstrap_full")
    emit.add_argument("--rtdata-name", default="dsl_bootstrap_get_rt_data_info")
    emit.add_argument("--pt-from-msg-name", default="dsl_bts_Pt_from_msg")
    emit.add_argument("--rotate-name", default="dsl_bts_Rotate")
    emit.add_argument("--relin-name", default="dsl_bts_Relinearize")
    emit.add_argument("--raise-level-name", default="dsl_bts_raise_level")
    emit.add_argument("--const-prefix", default="dsl_bts")
    emit.set_defaults(func=emit_body)

    shim = subparsers.add_parser("emit-shim")
    shim.add_argument("--output", required=True)
    shim.add_argument("--entry-name", default="dsl_bootstrap_full")
    shim.add_argument("--ctxparams-name", default="Get_extra_context_params")
    shim.add_argument("--rtdata-name", default="dsl_bootstrap_get_rt_data_info")
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
