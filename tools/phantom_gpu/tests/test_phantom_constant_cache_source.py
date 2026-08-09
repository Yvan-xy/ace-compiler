from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ABI = REPO_ROOT / "fhe-cmplr" / "rtlib" / "include" / "rt_phantom" / "rt_def.h"
API = REPO_ROOT / "fhe-cmplr" / "rtlib" / "include" / "rt_phantom" / "phantom_api.h"
WRAPPERS = (
    REPO_ROOT / "fhe-cmplr" / "rtlib" / "include" / "rt_phantom" / "rt_phantom.h"
)
ADAPTER = REPO_ROOT / "fhe-cmplr" / "rtlib" / "phantom" / "src" / "phantom_lib.cu"
HARNESS = (
    REPO_ROOT
    / "tools"
    / "phantom_gpu"
    / "harness"
    / "bootstrap_phantom_constants.cu"
)


def test_constant_manifest_and_runtime_abi_are_exposed() -> None:
    api = API.read_text(encoding="utf-8")
    wrappers = WRAPPERS.read_text(encoding="utf-8")

    assert "Get_phantom_constant_manifest()" in api
    assert "Phantom_encode_manifest_constant(PLAIN plain, uint32_t entry_id)" in api
    assert "Phantom_load_cached_constant(PLAIN plain, uint32_t entry_id)" in api
    assert "PHANTOM_SETUP_METRICS Phantom_get_setup_metrics()" in api
    assert "inline void Load_cached_plain(PLAIN plain, uint32_t entry_id)" in wrappers
    assert "Phantom_load_cached_constant(plain, entry_id)" in wrappers


def test_cache_key_binds_the_complete_provider_tuple() -> None:
    source = ADAPTER.read_text(encoding="utf-8")
    key = source[source.index("struct ConstantCacheKey") : source.index(
        "[[noreturn]] void Fail"
    )]

    for field in (
        "_constant_id",
        "_parameter_fingerprint",
        "_chain_index",
        "_raw_scale",
        "_element_type",
        "_slot_count",
    ):
        assert field in key
    assert "context_data.parms().parms_id()" in source
    assert "plain.parms_id() != key._parameter_fingerprint" in source


def test_cache_is_built_after_keys_and_owned_by_context() -> None:
    source = ADAPTER.read_text(encoding="utf-8")
    constructor = source[source.index("  PHANTOM_CONTEXT()\n") : source.index(
        "static void SynchronizeDevice"
    )]
    destructor = source[source.index("~PHANTOM_CONTEXT()") : source.index(
        "void PrepareInput"
    )]

    assert constructor.index("gen_publickey") < constructor.index(
        "BuildConstantCache()"
    )
    assert constructor.index("create_galois_keys_from_steps") < constructor.index(
        "BuildConstantCache()"
    )
    assert "std::map<ConstantCacheKey, Plaintext> _constant_cache" in source
    assert "_constant_cache.clear()" in destructor


def test_correctness_encode_and_immutable_cached_load_are_distinct() -> None:
    source = ADAPTER.read_text(encoding="utf-8")
    encode = source[source.index("void EncodeManifestConstant") : source.index(
        "void LoadCachedConstant"
    )]
    load = source[source.index("void LoadCachedConstant") : source.index(
        "PHANTOM_SETUP_METRICS SetupMetrics"
    )]

    assert "EncodeConstantPayload" in encode
    assert "_constant_cache.find(key)" in load
    assert "*plain = cached->second" in load
    assert "EncodeConstantPayload" not in load


def test_manifest_schemas_hashes_and_setup_metrics_are_validated() -> None:
    abi = ABI.read_text(encoding="utf-8")
    source = ADAPTER.read_text(encoding="utf-8")

    assert "constexpr uint32_t constant_schema_version = 1" in source
    assert "constexpr uint32_t context_schema_version = 1" in source
    assert "constexpr uint32_t resource_schema_version = 3" in source
    assert "IsSha256(_constants->_context_manifest_sha256)" in source
    assert "IsSha256(entry._payload_sha256)" in source
    assert "IsSha256(entry._cache_key_sha256)" in source
    assert "_context_and_key_setup_seconds" in source
    assert "_plaintext_cache_setup_seconds" in source
    assert "_context_and_key_device_bytes" in source
    assert "_plaintext_cache_device_bytes" in source
    assert "_plaintext_cache_logical_device_bytes" in abi
    assert "_plaintext_cache_logical_device_bytes" in source
    assert "_plaintext_cache_host_bytes" in source
    assert "_plaintext_cache_entries" in source


def test_cache_metrics_use_checked_exact_retained_allocation_counts() -> None:
    source = ADAPTER.read_text(encoding="utf-8")
    harness = HARNESS.read_text(encoding="utf-8")

    assert "cached.poly_modulus_degree()" in source
    assert "cached.coeff_modulus_size()" in source
    assert "logical device byte count overflowed" in source
    assert "host metadata byte count overflowed" in source
    assert "total host byte count overflowed" in source
    assert "entry._scale_degree <= 0" in source
    assert "metrics._plaintext_cache_logical_device_bytes ==" in harness
    assert '"logical_device_bytes"' in harness
    assert "metrics._plaintext_cache_device_bytes > 0" not in harness


def test_production_adapter_does_not_introduce_native_bts() -> None:
    source = ADAPTER.read_text(encoding="utf-8")
    for forbidden in (
        '#include "bootstrap.cuh"',
        "Bootstrapper",
        "Phantom_bootstrap(",
        "bootstrap_3(",
    ):
        assert forbidden not in source
