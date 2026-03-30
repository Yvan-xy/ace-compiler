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


ROTATE_INIT = "  Init_ciph_same_scale(&_pgen_rot_res_2, &ciph_0, 0);\n"
ROTATE_FAST_PATH = (
    ROTATE_INIT
    + "  if (rot_idx_1 == 0) {\n"
    + "    Copy_ciphertext(&_pgen_rot_res_2, &ciph_0);\n"
    + "    RTLIB_TM_END(20, rtm);\n"
    + "    return _pgen_rot_res_2;\n"
    + "  }\n"
)
CTX_PAT = re.compile(
    r"(static\s+CKKS_PARAMS\s+parm\s*=\s*\{\s*"
    r"LIB_ANT\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*"
    r"\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*)"
    r"(\d+)\s*,\s*\n\s*\{\s*([^}]*)\s*\}",
    re.S,
)


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


def _extract_rotation_indices(bootstrap_c: str) -> list[int]:
    rot_idxs: set[int] = set()
    for pattern in (
        r"\bRotate\s*\([^,]+,\s*(-?\d+)\s*\)",
        r"\bRotate_ciph\s*\([^,]+,\s*[^,]+,\s*(-?\d+)\s*\)",
    ):
        for match in re.finditer(pattern, bootstrap_c):
            rot_idxs.add(int(match.group(1)))

    deg_match = re.search(
        r"static\s+CKKS_PARAMS\s+parm\s*=\s*\{\s*LIB_ANT\s*,\s*(\d+)",
        bootstrap_c,
        re.S,
    )
    if deg_match and "Conjugate_ciph(" in bootstrap_c:
        ring_degree = int(deg_match.group(1))
        rot_idxs.add(2 * ring_degree - 1)

    return sorted(v for v in rot_idxs if v != 0)


def _patch_zero_rotate_fast_path(source: str) -> str:
    return source.replace(ROTATE_INIT, ROTATE_FAST_PATH, 1)


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


def patch_resnet_context(args: argparse.Namespace) -> int:
    bootstrap_path = Path(args.bootstrap_c)
    resnet_inc_path = Path(args.resnet_inc)

    bootstrap_c = bootstrap_path.read_text(encoding="utf-8")
    resnet_inc = resnet_inc_path.read_text(encoding="utf-8")

    match = CTX_PAT.search(resnet_inc)
    if match is None:
        raise SystemExit("failed to locate CKKS_PARAMS in resnet include")

    existing = []
    for token in re.split(r"[,\s]+", match.group(3).strip()):
        if token:
            existing.append(int(token))

    all_rot_idxs = sorted(set(existing) | set(_extract_rotation_indices(bootstrap_c)))
    rot_list = ", ".join(str(v) for v in all_rot_idxs)
    patched = CTX_PAT.sub(
        lambda mm: f"{mm.group(1)}{len(all_rot_idxs)}, \n    {{ {rot_list} }}",
        resnet_inc,
        count=1,
    )
    patched = _patch_zero_rotate_fast_path(patched)
    resnet_inc_path.write_text(patched, encoding="utf-8")
    return 0


def emit_body(args: argparse.Namespace) -> int:
    src_path = Path(args.bootstrap_c)
    dst_path = Path(args.output)

    skip = False
    brace_depth = 0
    out_lines: list[str] = []

    def starts_wrapper(line: str) -> bool:
        return line.startswith("CKKS_PARAMS* Get_context_params()")

    with src_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not skip and starts_wrapper(line):
                skip = True
                brace_depth = line.count("{") - line.count("}")
                continue
            if skip:
                brace_depth += line.count("{") - line.count("}")
                if brace_depth <= 0:
                    skip = False
                continue
            line = re.sub(r"\bbootstrap_full\b", args.entry_name, line)
            line = re.sub(r"\bGet_rt_data_info\b", args.rtdata_name, line)
            line = re.sub(r"\bPt_from_msg\b", args.pt_from_msg_name, line)
            line = re.sub(r"\bRotate\b", args.rotate_name, line)
            line = re.sub(r"\bRelinearize\b", args.relin_name, line)
            line = re.sub(r"\b(_cst_\d+)\b", rf"{args.const_prefix}\1", line)
            out_lines.append(line)

    body = _patch_zero_rotate_fast_path("".join(out_lines))
    pt_decl = (
        f'void* {args.pt_from_msg_name}(void* pt, uint32_t index, size_t len, '
        f'uint32_t scale, uint32_t level);\n'
    )
    if pt_decl not in body:
        include_line = '#include "rt_ant/rt_ant.h"\n'
        body = body.replace(include_line, include_line + "\n" + pt_decl, 1)

    dst_path.write_text(body, encoding="utf-8")
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

        CIPHERTEXT {args.entry_name}(CIPHERTEXT p0, CIPHERTEXT p1);
        RT_DATA_INFO* {args.rtdata_name}(void);
        PLAIN {args.pt_from_msg_name}(void* pt, uint32_t index, size_t len,
                                      uint32_t scale, uint32_t level);

        static unsigned long g_dsl_bts_call_counter = 0;

        static double dsl_bts_now_sec(void) {{
          struct timespec ts;
          clock_gettime(CLOCK_MONOTONIC, &ts);
          return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
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
          char* slot_buf;
          uint64_t slot_size;
          uint32_t slot_count;
          int initialized;
        }} DSL_BTS_PT_FILE;

        static DSL_BTS_PT_FILE g_dsl_bts_pt = {{-1, {{0}}, NULL, NULL, 0, 0, 0}};

        static void dsl_bts_pt_fini(void) {{
          if (!g_dsl_bts_pt.initialized) {{
            return;
          }}
          if (g_dsl_bts_pt.fd >= 0) {{
            close(g_dsl_bts_pt.fd);
          }}
          free(g_dsl_bts_pt.lut);
          free(g_dsl_bts_pt.slot_buf);
          g_dsl_bts_pt.fd = -1;
          g_dsl_bts_pt.lut = NULL;
          g_dsl_bts_pt.slot_buf = NULL;
          g_dsl_bts_pt.slot_size = 0;
          g_dsl_bts_pt.slot_count = 0;
          g_dsl_bts_pt.initialized = 0;
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
          g_dsl_bts_pt.slot_count = (uint32_t)g_dsl_bts_pt.hdr._ent_count;
          const char* env = getenv("PT_ENTRY_COUNT");
          if (env != NULL && env[0] != '\\0') {{
            unsigned long val = strtoul(env, NULL, 10);
            if (val > 0) {{
              g_dsl_bts_pt.slot_count = (uint32_t)val;
            }}
          }}
          g_dsl_bts_pt.slot_size = Max_plain_buffer_length();
          g_dsl_bts_pt.slot_buf =
              (char*)malloc(g_dsl_bts_pt.slot_size * g_dsl_bts_pt.slot_count);
          if (g_dsl_bts_pt.slot_buf == NULL) {{
            fprintf(stderr, "[{args.log_prefix}] failed to allocate bootstrap plaintext cache\\n");
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
          if (index >= g_dsl_bts_pt.hdr._ent_count) {{
            fprintf(stderr, "[{args.log_prefix}] bootstrap plaintext index out of range: %u\\n", index);
            fflush(stderr);
            abort();
          }}
          uint32_t slot = index % g_dsl_bts_pt.slot_count;
          char* slot_ptr = g_dsl_bts_pt.slot_buf + slot * g_dsl_bts_pt.slot_size;
          struct DATA_LUT_ENTRY* lut = &g_dsl_bts_pt.lut[index];
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

          CIPHERTEXT out = {args.entry_name}(in_copy, in_copy);
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

    patch = subparsers.add_parser("patch-resnet-context")
    patch.add_argument("--bootstrap-c", required=True)
    patch.add_argument("--resnet-inc", required=True)
    patch.set_defaults(func=patch_resnet_context)

    emit = subparsers.add_parser("emit-body")
    emit.add_argument("--bootstrap-c", required=True)
    emit.add_argument("--output", required=True)
    emit.add_argument("--entry-name", default="dsl_bootstrap_full")
    emit.add_argument("--rtdata-name", default="dsl_bootstrap_get_rt_data_info")
    emit.add_argument("--pt-from-msg-name", default="dsl_bts_Pt_from_msg")
    emit.add_argument("--rotate-name", default="dsl_bts_Rotate")
    emit.add_argument("--relin-name", default="dsl_bts_Relinearize")
    emit.add_argument("--const-prefix", default="dsl_bts")
    emit.set_defaults(func=emit_body)

    shim = subparsers.add_parser("emit-shim")
    shim.add_argument("--output", required=True)
    shim.add_argument("--entry-name", default="dsl_bootstrap_full")
    shim.add_argument("--rtdata-name", default="dsl_bootstrap_get_rt_data_info")
    shim.add_argument("--pt-from-msg-name", default="dsl_bts_Pt_from_msg")
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
