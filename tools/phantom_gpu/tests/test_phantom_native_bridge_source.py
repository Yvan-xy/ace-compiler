from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ABI = REPO_ROOT / "fhe-cmplr" / "rtlib" / "include" / "rt_phantom" / "rt_def.h"
API = REPO_ROOT / "fhe-cmplr" / "rtlib" / "include" / "rt_phantom" / "phantom_api.h"
ADAPTER = REPO_ROOT / "fhe-cmplr" / "rtlib" / "phantom" / "src" / "phantom_lib.cu"


def test_borrowed_runtime_view_exposes_only_provider_objects() -> None:
    abi = ABI.read_text(encoding="utf-8")
    runtime_name = abi.index("PHANTOM_BORROWED_RUNTIME")
    view = abi[abi.rindex("typedef struct {", 0, runtime_name) :]
    view = view[: view.index("PHANTOM_BORROWED_RUNTIME;")]

    for declaration in (
        "PhantomContext*",
        "PhantomCKKSEncoder*",
        "PhantomSecretKey*",
        "PhantomPublicKey*",
        "PhantomRelinKey*",
        "PhantomGaloisKey*",
    ):
        assert declaration in view
    assert "std::unique_ptr" not in view
    assert "Bootstrapper" not in view
    assert "valid only until Finalize_context()" in abi


def test_bridge_is_one_borrow_call_with_explicit_lifetime_contract() -> None:
    api = API.read_text(encoding="utf-8")

    assert api.count("PHANTOM_BORROWED_RUNTIME Phantom_borrow_runtime();") == 1
    assert "between Prepare_context() and Finalize_context()" in api
    assert "remain owned by the runtime" in api


def test_bridge_borrows_the_existing_singleton_without_allocating() -> None:
    source = ADAPTER.read_text(encoding="utf-8")
    member_start = source.index("PHANTOM_BORROWED_RUNTIME BorrowRuntime()")
    member_end = source.index("\n  }", member_start) + 4
    member = source[member_start:member_end]
    api_start = source.index("PHANTOM_BORROWED_RUNTIME Phantom_borrow_runtime()")
    api_end = source.index("\n}", api_start) + 2
    api = source[api_start:api_end]

    for owner in (
        "_context.get()",
        "_encoder.get()",
        "_secret_key.get()",
        "_public_key.get()",
        "_relin_key.get()",
        "_galois_key.get()",
    ):
        assert owner in member
    assert "new " not in member
    assert "PHANTOM_CONTEXT::Context()->BorrowRuntime()" in api
    assert "Initialize" not in api
