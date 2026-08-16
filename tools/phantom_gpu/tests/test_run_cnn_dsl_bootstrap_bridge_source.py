from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BRIDGE = (
    REPO_ROOT / "tools" / "phantom_gpu" / "harness" / "run_cnn_dsl_bootstrap_bridge.cu"
)
RUNNER = REPO_ROOT / "tools" / "phantom_gpu" / "run_cnn_dsl_bootstrap.sh"
IO_METADATA = (
    REPO_ROOT / "tools" / "phantom_gpu" / "harness" / "run_cnn_phantom_io_metadata.cc"
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _compact(value: str) -> str:
    return "".join(value.split())


def test_bridge_has_one_narrow_c_abi_and_cpp_generated_call() -> None:
    source = _source(BRIDGE)

    assert source.count('extern "C" int Ace_phantom_run_cnn_dsl_bootstrap(') == 1
    assert "PhantomCiphertext *output, PhantomCiphertext *input" in source
    assert "phantom::CKKSEvaluator *evaluator" in source
    assert "std::size_t logical_slots" in source
    assert "std::size_t stage" in source
    assert (
        source.count("extern CIPHERTEXT bootstrap_full(CIPHERTEXT, CIPHERTEXT);") == 1
    )
    assert 'extern "C" CIPHERTEXT bootstrap_full' not in source


def test_bridge_rejects_any_non_q32_full_packed_profile() -> None:
    source = _source(BRIDGE)

    for declaration in (
        "kPolynomialDegree = 65536",
        "kLogicalSlots = 32768",
        "kDataQCount = 32",
        "kSpecialPCount = 11",
        "kInputQCount = 1",
        "kOutputQCount = 17",
        "kQPartCount = 3",
        "kHammingWeight = 192",
        "kFirstModulusBits = 60",
        "kScalingModulusBits = 56",
    ):
        assert declaration in source
    for field in (
        "_schema_version",
        "_resource_schema_version",
        "_packing",
        "_poly_degree",
        "_logical_slots",
        "_data_q_count",
        "_data_q_bit_sizes",
        "_special_p_count",
        "_special_p_bit_sizes",
        "_input_level",
        "_q_part_count",
        "_hamming_weight",
        "_first_modulus_bits",
        "_scaling_modulus_bits",
    ):
        assert f"manifest->{field}" in source
    assert "PHANTOM_PACKING_FULL" in source
    assert "context.total_parm_size() == kDataQCount + 1" in source
    assert "kDataQCount + kSpecialPCount" in source
    assert "runtime._encoder->logical_slot_count() == kLogicalSlots" in source
    assert "runtime._encoder->physical_slot_count() == kLogicalSlots" in source


def test_manifest_and_provider_identity_precede_ciphertext_access() -> None:
    source = _source(BRIDGE)
    entry = source.index('extern "C" int Ace_phantom_run_cnn_dsl_bootstrap(')
    body = source[entry:]

    manifest = body.index("Get_phantom_context_manifest()")
    validate_manifest = body.index("ValidateCompilerManifest(manifest, logical_slots)")
    borrow = body.index("Phantom_borrow_runtime()")
    validate_runtime = body.index("ValidateBorrowedRuntime(runtime, *evaluator)")
    validate_input = body.index(
        'ValidateProviderCiphertext(*runtime._context, *input, "CNN input")'
    )
    copy_input = body.index("copy_ciphertext(*runtime._context, *input, owned_input)")

    assert manifest < validate_manifest < borrow < validate_runtime
    assert validate_runtime < validate_input < copy_input


def test_bridge_shares_context_and_relin_but_permits_cnn_galois_subset() -> None:
    source = _source(BRIDGE)

    assert "evaluator.context == runtime._context" in source
    assert "evaluator.relin_keys == runtime._relin_key" in source
    assert "evaluator.galois_keys->is_generated()" in source
    assert "evaluator.galois_keys->parms_id() == key_parms_id" in source
    assert "evaluator.galois_keys == runtime._galois_key" not in source
    assert source.count("evaluator.encoder.decode(") == 1
    assert source.count("evaluator.decryptor.decrypt(") == 1
    assert "evaluator.encryptor." not in source


def test_bridge_owns_registered_q1_inputs_and_synchronizes_generated_call() -> None:
    source = _compact(_source(BRIDGE))

    copy_input = source.index("copy_ciphertext(*runtime._context,*input,owned_input);")
    register_input = source.index("Register_ciph_lifetime(&owned_input);")
    switch_input = source.index("Mod_switch(&owned_input,&owned_input);")
    copy_aux = source.index("copy_ciphertext(*runtime._context,owned_input,auxiliary);")
    register_aux = source.index("Register_ciph_lifetime(&auxiliary);")
    pre_sync = source.index('CheckCuda(cudaDeviceSynchronize(),"pre-DSL-bootstrap')
    call = source.index(
        "PhantomCiphertextresult=bootstrap_full(owned_input,auxiliary);"
    )
    register_result = source.index("Register_ciph_lifetime(&result);")
    post_sync = source.index('CheckCuda(cudaDeviceSynchronize(),"post-DSL-bootstrap')

    assert copy_input < register_input < switch_input
    assert switch_input < copy_aux < register_aux < pre_sync
    assert pre_sync < call < register_result < post_sync


def test_bridge_validates_q17_metadata_before_moving_result() -> None:
    source = _source(BRIDGE)
    compact = _compact(source)

    for contract in (
        "active_q == kOutputQCount",
        "output->chain_index() == expected_chain",
        "Active_q_count(output)",
        "Level(output)",
        "Chain_index(output)",
        "Get_ciph_slots(output)",
        "Get_ciph_size(output) == 2",
        "Is_ciph_ntt(output)",
        "Sc_degree(output) == 1.0",
        "HasDegreeOneScale(raw_scale)",
    ):
        assert contract in source

    validate = compact.index("ValidateBootstrapOutput(&result,*runtime._context);")
    move = compact.index("*output=std::move(result);")
    register = compact.index("Register_ciph_lifetime(output);")
    assert validate < move < register
    assert "ACE_PHANTOM_RUN_CNN_DSL_BOOTSTRAP_ERRORstage=%zu" in compact


def test_bridge_does_not_encode_a_fractional_half_at_raw_scale_one() -> None:
    source = _source(BRIDGE)

    # Phantom's raw helper encodes at plaintext scale one.  A fractional
    # broadcast constant such as 0.5 is rounded to an integer coefficient and
    # therefore cannot implement an exact no-level real projection.
    assert "multiply_const_raw_inplace" not in source


def test_registered_locals_are_scope_cleaned_on_success_and_exception() -> None:
    source = _compact(_source(BRIDGE))

    assert "classRegisteredCipherCleanup" in source
    cleanup = source[source.index("voidCleanup()") :]
    cleanup = cleanup[: cleanup.index("private:")]
    assert cleanup.index("Zero_ciph(_value);") < cleanup.index("Free_ciph(_value);")
    for value in ("owned_input", "auxiliary", "result"):
        register = source.index(f"Register_ciph_lifetime(&{value});")
        guard = source.index(f"RegisteredCipherCleanup{value}_cleanup(&{value});")
        assert register < guard

    move = source.index("*output=std::move(result);")
    register_output = source.index("Register_ciph_lifetime(output);")
    retire_source = source.index("result_cleanup.Cleanup();")
    assert move < register_output < retire_source


def test_runner_constructs_native_evaluator_from_borrowed_provider_objects() -> None:
    source = _source(RUNNER)

    assert "Phantom_borrow_runtime()" in source
    assert 'python3 - "${RUN_CNN_SOURCE}"' in source
    assert '"runtime._secret_key->create_galois_keys_from_steps("' in source
    assert '"*runtime._context,cnn_rotation_basis,"' in source
    assert '"CKKSEvaluatorckks_evaluator("' in source
    assert '"runtime._context,runtime._public_key,runtime._secret_key,"' in source
    assert '"runtime._encoder,runtime._relin_key,&cnn_galois_keys,scale);"' in source
    assert "Ace_phantom_run_cnn_dsl_bootstrap" in source
    assert "run_cnn_dsl_bootstrap_bridge.cu" in source


def test_runner_requires_same_level_plaintext_batch_normalization() -> None:
    source = _source(RUNNER)

    assert '"voidace_matched_batch_norm("' in source
    assert (
        '"ckks_evaluator.encoder.encode(offset,value.scale(),encoded_offset);"'
        in source
    )


def test_runner_requires_identity_bound_clear_imag_resnet_helper() -> None:
    source = _source(RUNNER)

    assert source.count('"clear_imag": True,') == 2
    assert '[-1:] == ["--clear-imag"]' in source
    assert 'expanded.get("clear_imag") is True' in source
    assert (
        '"terminal-real-projection-and-ciphertext-self-add-chain"'
        in source
    )
    assert '"terminal-conjugate-real-projection"' in source
    assert 'projection.get("projected_component") == "real"' in source
    assert 'projection.get("caller_proof_required") is True' in source
    assert (
        "--generate is unavailable for ResNet until compile_only has a "
        "real-output clear-imag qualification profile"
        in source
    )
    assert '"encoded_offset,value.chain_index());"' in source
    assert (
        '"ckks_evaluator.evaluator.sub_plain_inplace(value,encoded_offset);"'
        in source
    )
    assert (
        'compact.count("ace_matched_batch_norm(cnn,cnn,bn_bias[stage]") != 3'
        in source
    )


def test_run_cnn_io_metadata_owns_the_exact_zero_io_abi() -> None:
    source = _compact(_source(IO_METADATA))

    assert source.count('extern"C"{') == 1
    assert source.count("intGet_input_count(){return0;}") == 1
    assert source.count("intGet_output_count(){return0;}") == 1
    assert source.count("DATA_SCHEME*Get_encode_scheme(int){returnnullptr;}") == 1
    assert source.count("DATA_SCHEME*Get_decode_scheme(int){returnnullptr;}") == 1


def test_runner_links_io_metadata_into_both_host_executables() -> None:
    source = _source(RUNNER)

    assert (
        'IO_METADATA_SOURCE="${SCRIPT_DIR}/harness/run_cnn_phantom_io_metadata.cc"'
        in source
    )
    assert 'require_file "${IO_METADATA_SOURCE}"' in source
    assert '"${IO_METADATA_SOURCE}" -o "${IO_METADATA_OBJECT}"' in source
    assert source.count('"${IO_METADATA_OBJECT}"') == 3
    assert (
        source.count(
            "for callback in Get_input_count Get_output_count Get_encode_scheme Get_decode_scheme"
        )
        == 2
    )
