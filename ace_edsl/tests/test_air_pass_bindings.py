"""Real AIR traversal, ownership, remapping, and transaction contracts."""

from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass

import pytest

from ace_bindings import air_builder
from ace_edsl.edsl.passes.framework import (
    AIRAttribute,
    AIRObjectId,
    AIRPassManager,
    AnalysisPass,
    AnalysisRequest,
    ModuleView,
    MutableModuleView,
    SourcePosition,
    TransformationPass,
)
from ace_edsl.edsl.passes.framework.api import analysis_success, transform_success


def _module(glob, invocation=1):
    return ModuleView(glob, (glob.air_pass_module_id, invocation))


def _fixture(*, helper=True):
    glob = air_builder.GlobScope()
    file_id = glob.register_file("air_pass_fixture.py")
    vector = glob.new_array_type([4], "f32")
    main = glob.new_func_with_param_types("main", vector, [vector])
    main_param = main.new_param("input", vector)
    main_container = main.container()
    main_container.set_loc(file_id, 11, 7)
    main_container.new_local("main_local", vector)
    main_container.new_preg("main_preg", vector)
    main_container.new_stp("main_preg", main_param)
    main_container.new_stid("main_local", main_param)
    loop = main_container.new_loop_begin_range(0, 2, 32)
    main_container.new_loop_index(loop)
    main_container.new_comment("inside")
    main_container.new_loop_end()
    main_container.new_floatconst_typed(1.25, glob.get_type("f32"))
    index = main_container.new_intconst_typed(1, glob.get_type("i32"))
    rolled = main_container.new_vec_roll(main_param, index, [0, 1])
    main_container.new_retv(rolled)

    helper_scope = None
    if helper:
        helper_scope = glob.new_func_with_param_types("helper", vector, [vector])
        helper_param = helper_scope.new_param("arg", vector)
        container = helper_scope.container()
        container.new_local("helper_local", vector)
        container.new_preg("helper_preg", vector)
        container.new_stp("helper_preg", helper_param)
        helper_loop = container.new_loop_begin_range(0, 2, 32)
        container.new_loop_index(helper_loop)
        container.new_comment("helper loop")
        container.new_loop_end()
        helper_index = container.new_intconst_typed(1, glob.get_type("i32"))
        helper_rolled = container.new_vec_roll(
            helper_param, helper_index, [0, 1]
        )
        container.new_stp("helper_preg", helper_rolled)
        container.new_retv(helper_rolled)
    assert glob.verify_ir()
    return glob, main, helper_scope


def _transaction(glob, invocation=1):
    transaction = glob.begin_air_pass_transaction()
    return transaction, MutableModuleView(
        glob, (glob.air_pass_module_id, invocation), transaction=transaction
    )


def _statement(function, opcode):
    return next(
        statement
        for block in function.blocks
        for statement in block.statements
        if statement.node.opcode == opcode
    )


def test_read_only_capability_and_deterministic_traversal_order():
    glob, _, _ = _fixture()
    module = _module(glob)
    assert [function.name for function in module.functions] == ["main", "helper"]
    assert [function.native_id for function in module.functions] == sorted(
        function.native_id for function in module.functions
    )
    assert [entry.native_id for entry in module.entries] == sorted(
        entry.native_id for entry in module.entries
    )
    main = module.functions[0]
    for values in (main.formals, main.locals, main.pregs):
        assert [value.native_id for value in values] == sorted(
            value.native_id for value in values
        )
    assert main.entry_statement.node.opcode == "func_entry"
    assert main.entry_block.parent_statement.id == main.entry_statement.id
    assert main.blocks[0] == main.entry_block
    assert [statement.node.opcode for statement in main.entry_block.statements] == [
        "stp",
        "st",
        "do_loop",
        "retv",
    ]
    assert main.entry_block.statements[0].node.preg is not None
    assert main.entry_block.statements[1].node.symbol is not None
    assert main.entry_block.statements[-1].node.return_value is not None
    loop = main.entry_block.statements[2].node
    rolled = main.entry_block.statements[-1].node.return_value
    assert [child.opcode for child in rolled.children] == ["ld", "intconst"]
    assert rolled.result_type.id == main.formals[0].type.id
    assert tuple(block.id for block in main.blocks) == (
        main.entry_block.id,
        loop.child_blocks[0].id,
    )
    raw_records = {
        tuple(record["key"]): record
        for record in glob.air_pass_snapshot()["objects"]
    }
    assert tuple(child.id.backend_key for child in rolled.children) == tuple(
        tuple(key) for key in raw_records[rolled.id.backend_key]["children"]
    )
    assert loop.iv.id.kind == "iv"
    assert [iv.native_id for iv in main.ivs] == [loop.iv.native_id]
    assert loop.iv.native_id not in {
        formal.native_id for formal in main.formals
    }
    assert loop.iv.native_id != next(
        local.native_id for local in main.locals if local.name == "main_local"
    )
    assert loop.child_blocks[0].parent_statement.id == loop.parent_statement.id
    assert not hasattr(module, "remove_function")
    assert not hasattr(main, "create_local")


def test_total_map_covers_every_source_kind_and_reuses_globals_explicitly():
    glob, _, _ = _fixture()
    source = _module(glob)
    transaction, candidate = _transaction(glob)
    remap = candidate.source_to_candidate
    source_keys = {
        tuple(record["key"]) for record in glob.air_pass_snapshot()["objects"]
    }
    assert {stable_id.backend_key for stable_id in remap} == source_keys
    assert remap[source.id].kind == "module"
    expected = {
        "module",
        "function",
        "entry",
        "formal",
        "local",
        "symbol",
        "preg",
        "iv",
        "block",
        "statement",
        "node",
        "type",
        "constant",
    }
    assert expected <= {stable_id.kind for stable_id in remap}
    for stable_id, mapped in remap.items():
        if stable_id.kind in {"module", "function", "entry", "formal", "local", "symbol", "preg", "iv", "type", "constant"}:
            assert stable_id.backend_key == mapped.backend_key
    with pytest.raises(ValueError, match="unmapped or wrong-kind"):
        transaction.remap_native_id("constant", None, 999999)
    main = source.functions[0]
    normal_local = next(local for local in main.locals if local.name == "main_local")
    with pytest.raises(ValueError, match="unmapped or wrong-kind"):
        transaction.remap_native_id(
            "iv", main.native_id, normal_local.native_id
        )
    forged_iv = AIRObjectId(
        "iv",
        main.id.module_id,
        main.formals[0].native_id,
        main.native_id,
        main.id.revision,
        main.id.generation,
    )
    with pytest.raises(RuntimeError, match="no longer present"):
        main.lookup(forged_iv)
    transaction.rollback()


def test_attribute_payload_and_source_position_round_trip_exactly():
    glob, _, _ = _fixture()
    transaction, candidate = _transaction(glob)
    main, helper = candidate.functions
    source = [
        statement
        for block in helper.blocks
        for statement in block.statements
        if statement.node.opcode == "stp"
        and statement.node.children[0].opcode == "roll"
    ][0]
    node = source.node.children[0]
    position = SourcePosition(1, 123, 17, 5, True, True)
    node.set_source_position(position)
    widths = {
        "bool": 1,
        "i8": 1,
        "i16": 2,
        "i32": 4,
        "i64": 8,
        "u8": 1,
        "u16": 2,
        "u32": 4,
        "u64": 8,
        "f32": 4,
        "f64": 8,
        "f80": 16,
        "f128": 16,
        "c32": 8,
        "c64": 16,
        "c80": 32,
        "c128": 32,
    }
    attributes = [AIRAttribute("text", "string", 0, b"raw payload")]
    attributes.extend(
        AIRAttribute(
            f"payload_{element_type}",
            element_type,
            2,
            bytes((index + offset) % 256 for offset in range(width * 2)),
        )
        for index, (element_type, width) in enumerate(widths.items())
    )
    for attribute in attributes:
        node.set_attribute(attribute)
    assert node.source_position == position
    expected_attributes = node.attributes
    assert set(attributes) <= set(expected_attributes)

    target = _statement(main, "retv")
    destination_preg = main.clone_preg(helper.pregs[0])
    cloned = target.clone_before(
        source,
        formal_map={helper.formals[0]: target.node.return_value},
        preg_map={helper.pregs[0]: destination_preg},
    )
    cloned_node = cloned.node.children[0]
    assert cloned_node.source_position == position
    assert cloned_node.attributes == expected_attributes
    assert candidate.verify()
    assert transaction.commit()
    committed = _module(glob, 2)
    cloned_node = next(
        statement.node.children[0]
        for block in committed.functions[0].blocks
        for statement in block.statements
        if statement.node.opcode == "stp"
        and statement.node.children[0].attributes == expected_attributes
    )
    assert cloned_node.source_position == position
    assert cloned_node.attributes == expected_attributes


@dataclass(frozen=True)
class _NoOp(TransformationPass):
    pass_id = "test.binding.noop"

    def run(self, unit, context):
        return transform_success()


@dataclass(frozen=True)
class _CreateLocal(TransformationPass):
    pass_id = "test.binding.create-local"
    name: str = "manager_local"

    def run(self, unit, context):
        unit.functions[0].create_local(self.name, unit.types[0])
        return transform_success(metrics={"locals_created": 1})


class _CountLocals(AnalysisPass):
    identity = "binding-local-count"
    pass_id = "analysis.binding-local-count"

    def __init__(self, counts):
        self._counts = counts

    def run(self, unit, context):
        self._counts.append(len(unit.functions[0].locals))
        return analysis_success(self._counts[-1])


def test_real_manager_transformation_commits_once_and_refreshes_public_scope():
    glob, _, _ = _fixture(helper=False)
    old_module = _module(glob)
    old_child = old_module.functions[0]
    counts = []
    analysis = _CountLocals(counts)
    before = (
        glob.dump(),
        glob.get_native_ptr(),
        glob.air_pass_revision,
        glob.air_pass_generation,
    )
    result = AIRPassManager().run(
        glob,
        (
            AnalysisRequest(analysis),
            _CreateLocal(),
            AnalysisRequest(analysis),
        ),
    )
    assert result.success and result.changed
    assert counts == [2, 3]
    assert not result.executions[0].cached
    assert result.executions[1].metrics["locals_created"] == 1
    assert result.executions[1].before_revision == 0
    assert result.executions[1].after_revision == 1
    assert not result.executions[2].cached
    assert glob.get_native_ptr() != before[1]
    assert (glob.air_pass_revision, glob.air_pass_generation) == (1, 1)
    assert glob.verify_ir()
    fresh = _module(glob, 2)
    assert "manager_local" in {local.name for local in fresh.functions[0].locals}
    assert glob.get_type("f32").to_string()
    with pytest.raises(RuntimeError, match="stale AIR view"):
        _ = old_child.name


def test_module_ids_and_unit_refs_stale_with_their_commit_or_rollback_generation():
    glob, _, _ = _fixture(helper=False)
    committed = _module(glob)
    committed_id = committed.id
    assert not hasattr(committed, "native_pointer")
    transaction, candidate = _transaction(glob)
    candidate_id = candidate.id
    candidate.functions[0].create_local("committed", candidate.types[0])
    assert transaction.commit()
    for use in (
        lambda: committed.id,
        lambda: committed.unit_ref,
        lambda: candidate.id,
        lambda: candidate.unit_ref,
    ):
        with pytest.raises(RuntimeError, match="stale AIR"):
            use()

    fresh = _module(glob, 2)
    forged = AIRObjectId(
        "module",
        fresh.id.module_id,
        fresh.id.native_id + 1,
        revision=fresh.id.revision,
        generation=fresh.id.generation,
    )
    with pytest.raises(ValueError, match="does not identify this module"):
        fresh.lookup(forged)
    assert committed_id.revision == 0 and candidate_id.revision == 0

    rollback, rolled = _transaction(glob, 3)
    _ = (rolled.id, rolled.unit_ref)
    rollback.rollback()
    for use in (lambda: rolled.id, lambda: rolled.unit_ref):
        with pytest.raises(RuntimeError, match="stale AIR transaction"):
            use()


def test_active_transaction_excludes_committed_authoring_and_native_drivers():
    glob, native_function, _ = _fixture(helper=False)
    native_container = native_function.container()
    native_type = glob.get_type("f32")
    compiler = air_builder.FheCompiler()
    assert compiler.init_with_glob(glob)
    before = (
        glob.dump(),
        glob.get_native_ptr(),
        glob.air_pass_revision,
        glob.air_pass_generation,
    )
    transaction, _ = _transaction(glob)
    operations = (
        lambda: glob.register_file("forbidden.py"),
        lambda: glob.new_array_type([2], "f32"),
        lambda: glob.new_func_with_param_types(
            "forbidden", native_type, [native_type]
        ),
        lambda: native_container.new_comment("forbidden"),
        lambda: native_function.new_param("forbidden", native_type),
        native_type.to_string,
        air_builder.Type.make_polynomial,
        lambda: glob.invalidate_air_pass_views(False),
        lambda: air_builder.run_ckks_driver(glob),
        lambda: air_builder.run_poly_driver(glob),
        compiler.pre_run,
        compiler.run,
        compiler.post_run,
        lambda: compiler.run_full_pipeline(glob),
    )
    for operation in operations:
        with pytest.raises(RuntimeError, match="transaction|locked"):
            operation()
    assert not transaction.dirty
    transaction.rollback()
    assert (
        glob.dump(),
        glob.get_native_ptr(),
        glob.air_pass_revision,
        glob.air_pass_generation,
    ) == before
    assert glob.verify_ir()


def test_noop_preserves_pointer_revision_generation_and_child_views():
    glob, _, _ = _fixture(helper=False)
    module = _module(glob)
    child = module.functions[0].entry_block.statements[-1].node
    before = (glob.get_native_ptr(), glob.air_pass_revision, glob.air_pass_generation)
    result = AIRPassManager().run(glob, (_NoOp(),))
    assert result.success and not result.changed
    assert (glob.get_native_ptr(), glob.air_pass_revision, glob.air_pass_generation) == before
    assert child.opcode == "retv"


def test_commit_invalidates_all_candidate_and_precommit_children_but_not_wrapper():
    glob, native_function, _ = _fixture(helper=False)
    native_container = native_function.container()
    native_node = native_function.new_param("cached", glob.get_type("f32"))
    native_type = glob.get_type("f32")
    committed = _module(glob)
    old_child = committed.functions[0]
    committed_function = committed.functions[0]
    committed_views = (
        committed,
        committed_function,
        committed_function.entries[0],
        committed_function.formals[0],
        committed_function.locals[0],
        committed_function.symbols[0],
        committed_function.pregs[0],
        committed_function.ivs[0],
        committed_function.blocks[0],
        committed_function.blocks[0].statements[0],
        committed_function.blocks[0].statements[0].node,
        committed.types[0],
        committed.constants[0],
    )
    before_pointer = glob.get_native_ptr()
    transaction, candidate = _transaction(glob)
    candidate_function = candidate.functions[0]
    candidate_views = (
        candidate,
        candidate_function,
        candidate_function.entries[0],
        candidate_function.formals[0],
        candidate_function.locals[0],
        candidate_function.symbols[0],
        candidate_function.pregs[0],
        candidate_function.ivs[0],
        candidate_function.blocks[0],
        candidate_function.blocks[0].statements[0],
        candidate_function.blocks[0].statements[0].node,
        candidate.types[0],
        candidate.constants[0],
    )
    escaped = candidate.functions[0].create_local("committed", candidate.types[0])
    assert transaction.commit()
    assert glob.get_native_ptr() != before_pointer
    assert glob.air_pass_revision == 1 and glob.air_pass_generation == 1
    assert glob.verify_ir()
    assert _module(glob, 2).functions
    for view in (*committed_views, *candidate_views):
        with pytest.raises(RuntimeError, match="stale AIR"):
            _ = view.id
    for use in (
        lambda: old_child.name,
        lambda: escaped.name,
        lambda: native_function.name,
        native_container.dump,
        native_node.opcode_name,
        native_type.to_string,
    ):
        with pytest.raises(RuntimeError, match="stale|expired|inactive"):
            use()


def test_rollback_invalidates_candidate_only_and_reuses_no_generation_token():
    glob, _, _ = _fixture(helper=False)
    committed = _module(glob)
    child = committed.functions[0]
    first_transaction, first = _transaction(glob)
    rolled_function = first.functions[0]
    rolled_views = (
        first,
        rolled_function,
        rolled_function.entries[0],
        rolled_function.formals[0],
        rolled_function.locals[0],
        rolled_function.symbols[0],
        rolled_function.pregs[0],
        rolled_function.ivs[0],
        rolled_function.blocks[0],
        rolled_function.blocks[0].statements[0],
        rolled_function.blocks[0].statements[0].node,
        first.types[0],
        first.constants[0],
    )
    first_generation = first_transaction.air_pass_generation
    first_transaction.rollback()
    for view in rolled_views:
        with pytest.raises(RuntimeError, match="stale AIR transaction"):
            _ = view.id
    assert child.name == "main"
    second_transaction, _ = _transaction(glob, 2)
    assert second_transaction.air_pass_generation != first_generation
    second_transaction.rollback()


def test_foreign_views_are_rejected_even_when_native_ids_collide():
    glob, _, _ = _fixture(helper=False)
    foreign, _, _ = _fixture(helper=False)
    transaction, candidate = _transaction(glob)
    foreign_transaction, foreign_candidate = _transaction(foreign)
    function = candidate.functions[0]
    with pytest.raises(ValueError, match="another module or transaction"):
        function.create_local("bad", foreign_candidate.types[0])
    with pytest.raises(ValueError, match="another module or transaction"):
        candidate.remove_function(foreign_candidate.functions[0])
    target = function.entry_block.statements[-1]
    with pytest.raises(ValueError, match="another module or transaction"):
        target.insert_before(foreign_candidate.functions[0].entry_block.statements[-1])
    assert not transaction.dirty
    transaction.rollback()
    foreign_transaction.rollback()


def test_insert_replace_erase_and_nested_parent_repair_commit_verified_air():
    glob, _, _ = _fixture(helper=False)
    transaction, candidate = _transaction(glob)
    function = candidate.functions[0]
    loop_statement = _statement(function, "do_loop")
    return_statement = _statement(function, "retv")
    clone = return_statement.insert_before(loop_statement)
    assert clone.node.child_blocks[0].parent_statement.id == clone.id
    before = return_statement.insert_before(function.entry_block.statements[0])
    after = return_statement.insert_after(function.entry_block.statements[0])
    replacement = before.replace_with(after)
    after.erase()
    with pytest.raises(RuntimeError, match="no longer present"):
        _ = after.native_id
    assert replacement.parent_block.id == function.entry_block.id
    assert candidate.verify()
    assert transaction.commit() and glob.verify_ir()


def test_verifier_failure_is_atomic_and_candidate_views_stale():
    glob, _, _ = _fixture(helper=False)
    pointer = glob.get_native_ptr()
    transaction, candidate = _transaction(glob)
    escaped = candidate.functions[0]
    transaction.force_verification_failure_for_testing()
    assert not transaction.commit()
    assert glob.get_native_ptr() == pointer
    assert glob.air_pass_revision == 0 and glob.verify_ir()
    with pytest.raises(RuntimeError, match="stale AIR transaction"):
        _ = escaped.native_id


def test_checked_edit_exception_leaves_committed_air_unchanged():
    glob, _, _ = _fixture(helper=False)
    pointer = glob.get_native_ptr()
    transaction, candidate = _transaction(glob)
    return_node = _statement(candidate.functions[0], "retv").node
    with pytest.raises(ValueError, match="result preg"):
        return_node.set_result_preg(candidate.functions[0].pregs[0])
    assert not transaction.dirty
    transaction.rollback()
    assert glob.get_native_ptr() == pointer and glob.air_pass_revision == 0


def test_type_mismatch_and_inconsistent_iv_maps_fail_before_structural_edit():
    glob, _, _ = _fixture()
    committed = (glob.dump(), glob.get_native_ptr(), glob.air_pass_revision)
    transaction, candidate = _transaction(glob)
    main, helper = candidate.functions
    main_loop = _statement(main, "do_loop").node
    main_local = next(local for local in main.locals if local.name == "main_local")
    with pytest.raises(ValueError, match="incompatible AIR type"):
        main_loop.set_iv(main_local)
    assert not transaction.dirty

    source_loop = _statement(helper, "do_loop")
    source_iv = source_loop.node.iv
    source_local = next(
        local for local in helper.locals if local.native_id == source_iv.native_id
    )
    first_destination = main.clone_iv(source_iv)
    second_destination = main.clone_iv(source_iv)
    before_failed_clone = transaction.air_pass_snapshot()
    with pytest.raises(ValueError, match="local and IV mappings disagree"):
        _statement(main, "retv").clone_before(
            source_loop,
            local_map={source_local: first_destination},
            iv_map={source_iv: second_destination},
        )
    assert transaction.air_pass_snapshot() == before_failed_clone
    transaction.rollback()
    assert (glob.dump(), glob.get_native_ptr(), glob.air_pass_revision) == committed


def test_complete_function_removal_drops_scope_entry_symbol_and_code():
    glob, _, _ = _fixture()
    transaction, candidate = _transaction(glob)
    helper = candidate.functions[1]
    helper_function_id = helper.native_id
    helper_entry_ids = {entry.native_id for entry in helper.entries}
    candidate.remove_function(helper)
    assert [function.name for function in candidate.functions] == ["main"]
    assert helper_entry_ids.isdisjoint(entry.native_id for entry in candidate.entries)
    assert transaction.commit() and glob.verify_ir()
    committed = _module(glob, 2)
    assert [function.name for function in committed.functions] == ["main"]
    snapshot = glob.air_pass_snapshot()
    assert ("function", None, helper_function_id) not in {
        tuple(record["key"]) for record in snapshot["objects"]
    }
    assert helper_entry_ids.isdisjoint(
        key[2]
        for record in snapshot["objects"]
        if (key := tuple(record["key"]))[0] == "entry"
    )
    assert "helper" not in glob.dump().split("FUNCTION TABLE", 1)[1]


def test_explicit_formal_local_preg_iv_mapped_cloning_across_functions():
    glob, _, _ = _fixture()
    transaction, candidate = _transaction(glob)
    main, helper = candidate.functions
    target = _statement(main, "retv")

    source_store = _statement(helper, "stp")
    created_preg = main.create_preg(main.formals[0].type)
    assert created_preg.id.kind == "preg"
    assert created_preg.type.id == main.formals[0].type.id
    destination_preg = main.clone_preg(helper.pregs[0])
    cloned_store = target.clone_before(
        source_store,
        formal_map={helper.formals[0]: target.node.return_value},
        preg_map={helper.pregs[0]: destination_preg},
    )
    assert cloned_store.node.preg.id == destination_preg.id

    source_local = next(
        local for local in helper.locals if local.name == "helper_local"
    )
    destination_local = main.clone_local(source_local)
    assert destination_local.id.kind == "local"
    assert destination_local.type.id == source_local.type.id

    source_loop = _statement(helper, "do_loop")
    source_iv = source_loop.node.iv
    source_local = next(
        local for local in helper.locals if local.native_id == source_iv.native_id
    )
    destination_iv = main.clone_iv(source_iv)
    destination_local = main.lookup(
        type(destination_iv.id)(
            "local",
            destination_iv.id.module_id,
            destination_iv.native_id,
            destination_iv.id.function_id,
            destination_iv.id.revision,
            destination_iv.id.generation,
        )
    )
    cloned_loop = target.clone_before(
        source_loop,
        local_map={source_local: destination_local},
        iv_map={source_iv: destination_iv},
    )
    assert cloned_loop.node.iv.native_id == destination_iv.native_id
    assert cloned_loop.node.child_blocks[0].parent_statement.id == cloned_loop.id
    assert candidate.verify()
    assert transaction.commit() and glob.verify_ir()


def test_repeated_commits_refresh_aliases_and_stale_every_retired_generation():
    glob, native_function, _ = _fixture(helper=False)
    native_function_ref = weakref.ref(native_function)
    pointers = [glob.get_native_ptr()]
    retired = []
    transient_type_refs = []
    for invocation in (1, 2):
        view = _module(glob, invocation)
        retained_type = glob.get_type("f32")
        transient_type = glob.get_type("f32")
        transient_type_refs.append(weakref.ref(transient_type))
        retired.append((view.functions[0], retained_type))
        transaction, candidate = _transaction(glob, invocation)
        candidate.functions[0].create_local(
            f"generation_{invocation}", candidate.types[0]
        )
        assert transaction.commit()
        pointers.append(glob.get_native_ptr())
        del transient_type
        if invocation == 1:
            del native_function
        gc.collect()
        assert transient_type_refs[-1]() is None
        assert glob.get_type("f32").to_string()
    assert all(left != right for left, right in zip(pointers, pointers[1:]))
    assert glob.air_pass_revision == 2 and glob.air_pass_generation == 2
    assert glob.verify_ir()
    assert native_function_ref() is None
    for function, native_type in retired:
        with pytest.raises(RuntimeError, match="stale|expired"):
            _ = function.name
        with pytest.raises(RuntimeError, match="expired"):
            native_type.to_string()
    assert glob.run_cpp_pass("vector2sihe", [])
    assert glob.verify_ir()
