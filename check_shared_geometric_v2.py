#!/usr/bin/env python3
"""Bounded correctness checks for the new shared-chemistry geometric model.

This file does not run on import. The explicit CLI uses one CPU thread and tiny
synthetic graph fixtures only: no molecule generation, no model fitting, no
test-set inference, and no changes to existing datasets/checkpoints/processes.
The optional prepared-cache check reads train records only.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


def _load_runtime():
    # Set these before importing numerical runtimes. The caller remains
    # responsible for scheduling this bounded check outside heavy work.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    global np, torch, Chem, Data, Batch, shared, model_module, gat_reference
    import numpy as np
    import torch
    from rdkit import Chem
    from torch_geometric.data import Data, Batch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    shared = importlib.import_module("shared_molecular_features_v2")
    model_module = importlib.import_module("geometric_bond_model_v2")
    gat_reference = importlib.import_module("train_gatv2_search")


def _encoded_edges(encoded):
    index = encoded.get("bond_edge_index", encoded.get("edge_index"))
    attributes = encoded.get("bond_edge_attr", encoded.get("edge_attr"))
    if index is None or attributes is None:
        raise AssertionError("Shared encoder did not return chemical edges and attributes")
    return np.asarray(index), np.asarray(attributes)


def _edge_dictionary(index, attributes):
    index, attributes = np.asarray(index), np.asarray(attributes)
    if index.shape != (2, len(attributes)):
        raise AssertionError("Chemical edge/attribute dimensions disagree")
    result = {}
    for offset in range(index.shape[1]):
        key = tuple(int(value) for value in index[:, offset])
        if key in result:
            raise AssertionError("Duplicate chemical edge in fixture/encoder")
        result[key] = attributes[offset]
    return result


def _manual_source(smiles, reverse=False):
    """Create correspondence fixtures, not a physically optimized conformer."""
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    conformer.Set3D(True)
    for index in range(molecule.GetNumAtoms()):
        # Exact decimal coordinates avoid SDF precision questions. Chemistry
        # comes from SMILES tags; no geometry/chirality inference is requested.
        conformer.SetAtomPosition(index, (0.37 * index, 0.23 * (index % 3), 0.19 * (index % 5)))
    molecule.AddConformer(conformer, assignId=True)
    if reverse:
        molecule = Chem.RenumberAtoms(molecule, list(reversed(range(molecule.GetNumAtoms()))))
    features = np.asarray([
        [atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetDegree(),
         int(atom.GetIsAromatic()), int(atom.GetHybridization()),
         int(atom.GetChiralTag()), atom.GetTotalNumHs(), int(atom.IsInRing())]
        for atom in molecule.GetAtoms()
    ], dtype=np.int16)
    positions = np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float32)
    edges, attributes = [], []
    for bond in molecule.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        values = [int(bond.GetBondType()), int(bond.GetIsAromatic()),
                  int(bond.GetIsConjugated()), int(bond.IsInRing()), int(bond.GetStereo())]
        edges.extend(((begin, end), (end, begin)))
        attributes.extend((values, values))
    index = np.asarray(edges, dtype=np.int64).reshape(-1, 2).T
    edge_features = np.asarray(attributes, dtype=np.int16).reshape(-1, 5)
    return molecule, features, positions, index, edge_features


def _fixture_graph(smiles="CCO"):
    encoded = shared.encode_smiles(smiles)
    z, x = np.asarray(encoded["z"]), np.asarray(encoded["x"])
    if len(z) != 3:
        raise AssertionError("Tiny model fixture expects three atoms")
    bonds, attributes = _encoded_edges(encoded)
    positions = torch.tensor([[0., 0., 0.], [1.4, 0., 0.], [2.1, 1.2, .3]], dtype=torch.float32)
    spatial = torch.tensor([(source, target) for target in range(3)
                            for source in range(3) if source != target], dtype=torch.long).T.contiguous()
    return Data(z=torch.tensor(z, dtype=torch.long), x=torch.tensor(x, dtype=torch.float32),
                pos=positions, bond_edge_index=torch.tensor(bonds, dtype=torch.long),
                bond_edge_attr=torch.tensor(attributes, dtype=torch.float32),
                spatial_edge_index=spatial, num_nodes=len(z), y=torch.tensor([0.]))


def _model(variant="angular"):
    torch.manual_seed(20261110)
    result = model_module.BondGeometryGNN(hidden=16, interactions=2, radial_basis=8,
        angular_basis=4, cutoff=4.5, dropout=0., geometry_variant=variant)
    result.eval()
    return result


class SharedChemistryChecks(unittest.TestCase):
    def test_exact_features_against_existing_gat_encoder(self):
        molecules = ["CCO", "CC(=O)O", "c1ccncc1", "C1CCCCC1", "C[C@H](O)F",
                     "C[C@@H](O)F", "F/C=C/F", r"F/C=C\F", "[NH4+]", "[Na+].[O-]C(=O)C"]
        for smiles in molecules:
            with self.subTest(smiles=smiles):
                encoded = shared.encode_smiles(smiles)
                reference = gat_reference.molecular_features(smiles)
                np.testing.assert_array_equal(encoded["z"], reference["z"].numpy())
                np.testing.assert_array_equal(encoded["x"], reference["x"].numpy())
                actual = _edge_dictionary(*_encoded_edges(encoded))
                expected = _edge_dictionary(reference["edge_index"].numpy(), reference["edge_attr"].numpy())
                self.assertEqual(set(actual), set(expected))
                for edge in expected:
                    np.testing.assert_array_equal(actual[edge], expected[edge])

    def test_attached_hydrogens_and_heavy_atom_degree(self):
        encoded = shared.encode_smiles("CCO")
        np.testing.assert_allclose(encoded["x"][:, 2], [.75, .5, .25], atol=1e-7)
        np.testing.assert_allclose(encoded["x"][:, 0], [1/6, 2/6, 1/6], atol=1e-7)

    def test_permuted_source_alignment_preserves_identity_and_stereo(self):
        for smiles in ["CCO", "N[C@@H](C)C(=O)O", "F/C=C/F", r"F/C=C\F"]:
            with self.subTest(smiles=smiles):
                molecule, features, positions, edges, attributes = _manual_source(smiles, reverse=True)
                actual = shared.aligned_features(smiles, molecule, features, positions,
                    source_edges=edges, source_bond_features=attributes)
                reference = shared.encode_smiles(smiles)
                mapping = np.asarray(actual["canonical_to_heavy"], dtype=np.int64)
                self.assertEqual(sorted(mapping.tolist()), list(range(len(mapping))))
                np.testing.assert_array_equal(actual["z"], features[features[:, 0] > 1, 0])
                np.testing.assert_array_equal(np.asarray(actual["x"])[mapping], reference["x"])
                canonical_edges, canonical_attributes = _encoded_edges(reference)
                expected = _edge_dictionary(mapping[canonical_edges], canonical_attributes)
                actual_edges = _edge_dictionary(actual["bond_edge_index"], actual["bond_edge_attr"])
                self.assertEqual(set(actual_edges), set(expected))
                for edge in expected:
                    np.testing.assert_array_equal(actual_edges[edge], expected[edge])

    def test_alignment_rejects_coordinate_or_element_mismatch(self):
        molecule, features, positions, _, _ = _manual_source("CCO", reverse=True)
        shifted = positions.copy()
        shifted[0, 0] += .01
        with self.assertRaises(ValueError):
            shared.aligned_features("CCO", molecule, features, shifted)
        wrong = features.copy()
        wrong[0, 0] = 8 if wrong[0, 0] != 8 else 6
        with self.assertRaises(ValueError):
            shared.aligned_features("CCO", molecule, wrong, positions)

    def test_alignment_rejects_opposite_chirality(self):
        molecule, features, positions, _, _ = _manual_source("N[C@@H](C)C(=O)O", reverse=True)
        with self.assertRaises(ValueError):
            shared.aligned_features("N[C@H](C)C(=O)O", molecule, features, positions)


class SharedModelChecks(unittest.TestCase):
    def test_all_variants_have_finite_graph_outputs(self):
        data = _fixture_graph()
        for variant in ("none", "distance", "angular"):
            with self.subTest(variant=variant), torch.no_grad():
                output = _model(variant)(data)
                self.assertEqual(tuple(output.shape), (1,))
                self.assertTrue(bool(torch.isfinite(output).all()))

    def test_isolated_atom_and_empty_edge_relations_are_finite(self):
        encoded = shared.encode_smiles("[Na+]")
        bonds, attributes = _encoded_edges(encoded)
        data = Data(z=torch.tensor(encoded["z"], dtype=torch.long),
            x=torch.tensor(encoded["x"], dtype=torch.float32), pos=torch.zeros((1, 3)),
            bond_edge_index=torch.tensor(bonds, dtype=torch.long),
            bond_edge_attr=torch.tensor(attributes, dtype=torch.float32),
            spatial_edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=1)
        for variant in ("none", "distance", "angular"):
            with self.subTest(variant=variant), torch.no_grad():
                output = _model(variant)(data)
                self.assertEqual(tuple(output.shape), (1,))
                self.assertTrue(bool(torch.isfinite(output).all()))

    def test_bond_attributes_have_nonzero_gradient_and_change_output(self):
        for variant in ("none", "angular"):
            with self.subTest(variant=variant):
                data = _fixture_graph()
                data.bond_edge_attr = data.bond_edge_attr.clone().requires_grad_(True)
                model = _model(variant)
                output = model(data)
                output.sum().backward()
                gradient = data.bond_edge_attr.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(bool(torch.isfinite(gradient).all()))
                self.assertGreater(float(gradient.abs().sum()), 1e-10)
                changed = data.clone()
                changed.bond_edge_attr = data.bond_edge_attr.detach().clone()
                # Change one bidirectional bond from single to double while
                # retaining valid chemical-type indicators in both directions.
                pair = set(map(int, data.bond_edge_index[:, 0].tolist()))
                for column in range(data.bond_edge_index.shape[1]):
                    if set(map(int, data.bond_edge_index[:, column].tolist())) == pair:
                        changed.bond_edge_attr[column, :5] = 0
                        changed.bond_edge_attr[column, 1] = 1
                with torch.no_grad():
                    difference = (model(changed) - output.detach()).abs().max()
                self.assertGreater(float(difference), 1e-8)

    def test_spatial_nonbonded_neighbor_is_separate_and_effective(self):
        data = _fixture_graph()
        chemical = {tuple(edge) for edge in data.bond_edge_index.T.tolist()}
        spatial = {tuple(edge) for edge in data.spatial_edge_index.T.tolist()}
        nonbonded = spatial - chemical
        self.assertTrue(nonbonded)
        selected = next(iter(nonbonded))
        keep = [column for column, edge in enumerate(data.spatial_edge_index.T.tolist())
                if tuple(edge) != selected]
        changed = data.clone()
        changed.spatial_edge_index = data.spatial_edge_index[:, keep]
        np.testing.assert_array_equal(changed.bond_edge_index.numpy(), data.bond_edge_index.numpy())
        np.testing.assert_array_equal(changed.bond_edge_attr.numpy(), data.bond_edge_attr.numpy())
        with torch.no_grad():
            distance_model = _model("distance")
            self.assertGreater(float((distance_model(data) - distance_model(changed)).abs().max()), 1e-8)
            none_model = _model("none")
            torch.testing.assert_close(none_model(data), none_model(changed), rtol=0., atol=0.)

    def test_rotation_translation_invariance(self):
        data = _fixture_graph()
        angle = .73
        rotation = torch.tensor([[math.cos(angle), -math.sin(angle), 0.],
                                 [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]], dtype=torch.float32)
        moved = data.clone()
        moved.pos = data.pos @ rotation.T + torch.tensor([2., -3., .5])
        for variant in ("none", "distance", "angular"):
            with self.subTest(variant=variant), torch.no_grad():
                model = _model(variant)
                torch.testing.assert_close(model(data), model(moved), rtol=2e-5, atol=2e-5)

    def test_atom_permutation_invariance(self):
        data = _fixture_graph()
        order = torch.tensor([2, 0, 1], dtype=torch.long)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(len(order))
        moved = data.clone()
        for key in ("z", "x", "pos"):
            moved[key] = data[key][order]
        moved.bond_edge_index = inverse[data.bond_edge_index]
        moved.spatial_edge_index = inverse[data.spatial_edge_index]
        for variant in ("none", "distance", "angular"):
            with self.subTest(variant=variant), torch.no_grad():
                model = _model(variant)
                torch.testing.assert_close(model(data), model(moved), rtol=2e-5, atol=2e-5)

    def test_pyg_batch_offsets_and_predictions_match_individual_graphs(self):
        first, second = _fixture_graph(), _fixture_graph("CCN")
        batch = Batch.from_data_list([first, second])
        for key in ("bond_edge_index", "spatial_edge_index"):
            index = batch[key]
            self.assertTrue(bool((batch.batch[index[0]] == batch.batch[index[1]]).all()))
            self.assertTrue(bool((index[:, first[key].shape[1]:] >= first.num_nodes).all()))
        for variant in ("none", "distance", "angular"):
            with self.subTest(variant=variant), torch.no_grad():
                model = _model(variant)
                individual = torch.cat([model(first), model(second)])
                torch.testing.assert_close(model(batch), individual, rtol=2e-5, atol=2e-5)

    def test_cross_molecule_chemical_and_spatial_edges_are_rejected(self):
        data = _fixture_graph()
        data.batch = torch.tensor([0, 0, 1], dtype=torch.long)
        with self.assertRaises(ValueError):
            _model("none")(data)
        data.bond_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data.bond_edge_attr = data.bond_edge_attr[:2].clone()
        with self.assertRaises(ValueError):
            _model("angular")(data)

    def test_weights_round_trip_without_prediction_change(self):
        data, original = _fixture_graph(), _model("angular")
        buffer = io.BytesIO()
        torch.save(original.state_dict(), buffer)
        buffer.seek(0)
        restored = _model("angular")
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        with torch.no_grad():
            torch.testing.assert_close(original(data), restored(data), rtol=0., atol=0.)


class PreparedTrainCacheChecks(unittest.TestCase):
    index_path = None
    summary_path = None
    trainer_module = "train_shared_geometric_v2"

    def test_tiny_sidecar_and_original_coordinate_cache_integration(self):
        trainer = importlib.import_module(self.trainer_module)
        examples = [_fixture_graph(), _fixture_graph("CCN")]
        with tempfile.TemporaryDirectory(prefix="shared_v2_cache_check_") as temporary:
            temporary = Path(temporary)
            geometry, chemistry = temporary / "geometry", temporary / "chemistry"
            geometry.mkdir()
            chemistry.mkdir()
            source = str(temporary / "synthetic_original_train.npz")
            source_ids = np.asarray(["fixture_a", "fixture_b"])
            molecule_ids = np.asarray(["mol_fixture_a", "mol_fixture_b"])
            atom_ptr = np.asarray([0, 3, 6], dtype=np.int64)
            spatial_ptr = np.asarray([0, 6, 12], dtype=np.int64)
            bond_ptr = np.asarray([0, 4, 8], dtype=np.int64)
            legacy_atoms = np.zeros((6, 8), dtype=np.int16)
            legacy_atoms[:, 0] = np.concatenate([graph.z.numpy() for graph in examples])
            spatial = np.concatenate([graph.spatial_edge_index.numpy() + 3 * index
                                      for index, graph in enumerate(examples)], axis=1)
            covalent = np.concatenate([graph.bond_edge_index.numpy() + 3 * index
                                       for index, graph in enumerate(examples)], axis=1)
            original_arrays = {
                "atom_features": legacy_atoms,
                "positions_angstrom": np.concatenate([graph.pos.numpy() for graph in examples]),
                "atom_ptr": atom_ptr, "edge_ptr": spatial_ptr, "edge_index": spatial,
                "source_id": source_ids, "molecule_id": molecule_ids,
                "label": np.asarray([0, 1]), "split": np.asarray(["train", "train"]),
                "optimization_converged": np.asarray([True, True]),
            }
            chemical_arrays = {
                "z": legacy_atoms[:, 0].astype(np.int64),
                "x": np.concatenate([graph.x.numpy() for graph in examples]),
                "atom_ptr": atom_ptr, "bond_ptr": bond_ptr,
                "bond_edge_index": covalent,
                "bond_edge_attr": np.concatenate([graph.bond_edge_attr.numpy() for graph in examples]),
                "selected_mask": np.asarray([True, True]),
                "source_id": source_ids, "molecule_id": molecule_ids,
            }
            for folder, arrays in ((geometry, original_arrays), (chemistry, chemical_arrays)):
                for name, values in arrays.items():
                    np.save(folder / (name + ".npy"), values, allow_pickle=False)
            index_path = temporary / "index.sqlite"
            connection = sqlite3.connect(index_path)
            try:
                connection.executescript("""CREATE TABLE records(split TEXT NOT NULL,local_index INTEGER NOT NULL,
                  source_id TEXT UNIQUE NOT NULL,molecule_id TEXT NOT NULL,graph_file TEXT NOT NULL,graph_number INTEGER NOT NULL,
                  label INTEGER NOT NULL,sample_weight REAL NOT NULL,method TEXT,category TEXT,subcategory TEXT,
                  optimization_converged INTEGER NOT NULL,PRIMARY KEY(split,local_index));""")
                for index in range(2):
                    connection.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        ("train", index, str(source_ids[index]), str(molecule_ids[index]), source, index,
                         index, 1., "M", "fixture", "fixture", 1))
                connection.commit()
            finally:
                connection.close()
            summary = {"counts": {"train": 2, "val": 0},
                "data_spec": {"source_files": {source: {"cache_dir": str(geometry)}}},
                "shared_features": {"complete": True, "sources": {source: {"directory": str(chemistry)}}}}
            dataset = trainer.SharedGeometryDataset(index_path, summary, "train")
            try:
                loaded = [dataset[0], dataset[1]]
                self.assertEqual(len(dataset), 2)
                for original, observed in zip(examples, loaded):
                    torch.testing.assert_close(observed.x, original.x, rtol=0., atol=0.)
                    torch.testing.assert_close(observed.pos, original.pos, rtol=0., atol=0.)
                    torch.testing.assert_close(observed.bond_edge_index, original.bond_edge_index, rtol=0., atol=0.)
                    torch.testing.assert_close(observed.bond_edge_attr, original.bond_edge_attr, rtol=0., atol=0.)
                    torch.testing.assert_close(observed.spatial_edge_index, original.spatial_edge_index, rtol=0., atol=0.)
                    self.assertTrue(bool((observed.x[:, 2] > 0).any()))
                with torch.no_grad():
                    output = _model("angular")(Batch.from_data_list(loaded))
                self.assertEqual(tuple(output.shape), (2,))
                self.assertTrue(bool(torch.isfinite(output).all()))
                with self.assertRaises(ValueError):
                    trainer.SharedGeometryDataset(index_path, summary, "test")
            finally:
                dataset.close()

    def test_prepared_train_records_and_batch_are_usable(self):
        if self.index_path is None or self.summary_path is None:
            self.skipTest("No optional prepared train-cache fixture supplied")
        trainer = importlib.import_module(self.trainer_module)
        summary = json.loads(Path(self.summary_path).read_text())
        dataset = trainer.SharedGeometryDataset(Path(self.index_path), summary, "train")
        try:
            self.assertGreaterEqual(len(dataset), 2)
            examples = [dataset[0], dataset[1]]
            for data in examples:
                self.assertEqual(data.x.shape[1], 21)
                self.assertEqual(data.bond_edge_attr.shape[1], 14)
                self.assertTrue(bool(torch.isfinite(data.pos).all()))
            batch = Batch.from_data_list(examples)
            with torch.no_grad():
                output = _model("angular")(batch)
            self.assertEqual(tuple(output.shape), (2,))
            self.assertTrue(bool(torch.isfinite(output).all()))
        finally:
            dataset.close()


class TrainerSafetyChecks(unittest.TestCase):
    def test_historical_nonempty_output_is_refused_without_mutation(self):
        trainer = importlib.import_module("train_shared_geometric_v2")
        with tempfile.TemporaryDirectory(prefix="shared_v2_output_check_") as temporary:
            out = Path(temporary) / "historical_run"
            out.mkdir()
            (out / "best.pt").write_bytes(b"historical model bytes\x00\x01")
            (out / "summary.json").write_bytes(b'{"historical":true}\n')
            original = {path.name: path.read_bytes() for path in out.iterdir()}
            args = SimpleNamespace(out=str(out), resume=None, prepare_only=False)
            with self.assertRaises(ValueError):
                trainer.preflight_out(args)
            self.assertEqual({path.name: path.read_bytes() for path in out.iterdir()}, original)

    def test_process_local_engine_hooks_restore_after_exception(self):
        trainer = importlib.import_module("train_shared_geometric_v2")
        keys = ("DistanceGNN", "GeometryDataset", "model_config", "locked_config", "make_loader")
        original = {key: getattr(trainer.engine, key) for key in keys}
        with self.assertRaisesRegex(RuntimeError, "intentional fixture exception"):
            with trainer.bound_shared_engine():
                self.assertIs(trainer.engine.DistanceGNN, model_module.BondGeometryGNN)
                self.assertIs(trainer.engine.GeometryDataset, trainer.SharedGeometryDataset)
                raise RuntimeError("intentional fixture exception")
        for key in keys:
            self.assertIs(getattr(trainer.engine, key), original[key])

    def test_exclusive_compute_rejects_nested_owner_in_temporary_workspace(self):
        resources = importlib.import_module("shared_geometry_resources_v2")
        with tempfile.TemporaryDirectory(prefix="shared_v2_lease_check_") as temporary:
            workspace = Path(temporary)
            with resources.exclusive_compute(workspace):
                with self.assertRaises(RuntimeError):
                    with resources.exclusive_compute(workspace):
                        self.fail("A competing lease unexpectedly acquired the same compute lock")
            # A failed competitor must not retain ownership after the first
            # lease exits. This path belongs solely to this temporary fixture.
            with resources.exclusive_compute(workspace):
                self.assertTrue((workspace / "molecule_bank_1m" / "bank.chemistry.lock").is_file())

    def test_partial_feature_preparation_refuses_before_fit(self):
        trainer = importlib.import_module("train_shared_geometric_v2")

        class DummyBudget:
            def __init__(self, *args, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def interval(self, *args, **kwargs):
                return nullcontext()

        with tempfile.TemporaryDirectory(prefix="shared_v2_partial_check_") as temporary:
            temporary = Path(temporary)
            out, index = temporary / "new_run", temporary / "unused_index.sqlite"
            original = {"cohort_sha256": trainer.COHORT_SHA256,
                        "counts": {"train": 161157, "val": 13756, "test": 16788}}
            with (
                mock.patch.object(trainer.resources, "exclusive_compute", return_value=nullcontext()),
                mock.patch.object(trainer.resources, "ComputeBudget", DummyBudget),
                mock.patch.object(trainer.engine, "seed_everything"),
                mock.patch.object(trainer.features, "prepare_shared_features", return_value={"complete": False}),
                mock.patch.object(trainer.engine, "prepare_data", return_value=(index, original)),
                mock.patch.object(trainer.engine, "fit", side_effect=AssertionError("Partial features reached fit")) as fit,
            ):
                with self.assertRaisesRegex(ValueError, "Partial feature preparation"):
                    trainer.main(["--out", str(out), "--threads", "1", "--device", "cpu", "--no-test",
                                  "--root", str(temporary / "unused_geometry"),
                                  "--graph-index", str(temporary / "unused_graph_index.csv"),
                                  "--cache-root", str(temporary / "unused_cache"),
                                  "--shared-feature-cache", str(temporary / "unused_sidecars")])
                fit.assert_not_called()
            self.assertFalse((out / "best.pt").exists())
            self.assertFalse((out / "last.pt").exists())
            self.assertFalse(index.exists())

    def test_resume_accepts_coherent_pair_and_rejects_foreign_or_inconsistent_best(self):
        trainer = importlib.import_module("train_shared_geometric_v2")
        expected = {"protocol_version": "fixture_v2", "cohort_sha256": "fixture_cohort"}
        with tempfile.TemporaryDirectory(prefix="shared_v2_resume_check_") as temporary:
            out = Path(temporary)
            last_path, best_path = out / "last.pt", out / "best.pt"
            last = {"configuration": expected, "epoch": 4, "best_epoch": 2,
                    "best_score": .8, "best_validation_f1": .8, "best_threshold": .4}
            best = {**last, "epoch": 2}
            torch.save(last, last_path)
            torch.save(best, best_path)
            args = SimpleNamespace(out=str(out), resume=str(last_path))
            with mock.patch.object(trainer, "locked_config", return_value=expected):
                original_last, original_best = last_path.read_bytes(), best_path.read_bytes()
                trainer.validate_resume_checkpoints(args, {}, torch.device("cpu"))
                self.assertEqual(last_path.read_bytes(), original_last)
                self.assertEqual(best_path.read_bytes(), original_best)
                bad_cases = [
                    {**best, "configuration": {**expected, "cohort_sha256": "foreign_cohort"}},
                    {**best, "epoch": 3},
                    {**best, "best_validation_f1": .79},
                ]
                for bad in bad_cases:
                    with self.subTest(best_checkpoint=bad):
                        torch.save(bad, best_path)
                        original_bad = best_path.read_bytes()
                        with self.assertRaises(ValueError):
                            trainer.validate_resume_checkpoints(args, {}, torch.device("cpu"))
                        self.assertEqual(last_path.read_bytes(), original_last)
                        self.assertEqual(best_path.read_bytes(), original_bad)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--prepared-index", type=Path)
    parser.add_argument("--prepared-summary", type=Path)
    parser.add_argument("--trainer-module", default="train_shared_geometric_v2")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if bool(args.prepared_index) != bool(args.prepared_summary):
        parser.error("Pass both --prepared-index and --prepared-summary")
    _load_runtime()
    loader, suite = unittest.TestLoader(), unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(SharedChemistryChecks))
    if not args.features_only:
        suite.addTests(loader.loadTestsFromTestCase(SharedModelChecks))
        PreparedTrainCacheChecks.index_path = args.prepared_index
        PreparedTrainCacheChecks.summary_path = args.prepared_summary
        PreparedTrainCacheChecks.trainer_module = args.trainer_module
        suite.addTests(loader.loadTestsFromTestCase(PreparedTrainCacheChecks))
        suite.addTests(loader.loadTestsFromTestCase(TrainerSafetyChecks))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = {"status": "passed" if result.wasSuccessful() else "failed", "checks_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
              "cpu_threads": 1, "training_performed": False, "dataset_test_split_accessed": False,
              "prepared_train_cache_checked": bool(args.prepared_index),
              "note": "Tiny synthetic fixtures; no conformer generation or trained-model test inference."}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
