"""
Unit tests for utils.persistence module.

Tests save/load/merge utilities with hybrid serialization.
"""

import pytest
import json
import pickle
from pathlib import Path
from src.utils.persistence import PersistenceManager, VectorStoreLoader


class TestPersistenceManagerJSON:
    """Test JSON save/load functionality."""

    def test_save_json(self, tmp_path):
        """Test saving JSON data."""
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        file_path = tmp_path / "test.json"

        PersistenceManager.save_json(file_path, data)

        # Verify file exists
        assert file_path.exists()

        # Verify content
        with open(file_path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_load_json(self, tmp_path):
        """Test loading JSON data."""
        data = {"key": "value", "nested": {"a": 1, "b": 2}}
        file_path = tmp_path / "test.json"

        # Create file
        with open(file_path, "w") as f:
            json.dump(data, f)

        # Load and verify
        loaded = PersistenceManager.load_json(file_path)
        assert loaded == data

    def test_save_json_with_indent(self, tmp_path):
        """Test JSON saved with indentation (human-readable)."""
        data = {"key": "value"}
        file_path = tmp_path / "test.json"

        PersistenceManager.save_json(file_path, data)

        # Verify indentation
        with open(file_path) as f:
            content = f.read()
        assert "{\n" in content  # Has newlines (indented)

    def test_save_json_creates_parent_dirs(self, tmp_path):
        """Test JSON save creates parent directories."""
        file_path = tmp_path / "nested" / "dir" / "test.json"
        data = {"key": "value"}

        PersistenceManager.save_json(file_path, data)

        assert file_path.exists()
        assert file_path.parent.exists()


class TestPersistenceManagerPickle:
    """Test Pickle save/load functionality."""

    def test_save_pickle(self, tmp_path):
        """Test saving pickle data."""
        data = {"key": "value", "list": [1, 2, 3], "nested": {"a": 1}}
        file_path = tmp_path / "test.pkl"

        PersistenceManager.save_pickle(file_path, data)

        # Verify file exists
        assert file_path.exists()

        # Verify content
        with open(file_path, "rb") as f:
            loaded = pickle.load(f)
        assert loaded == data

    def test_load_pickle(self, tmp_path):
        """Test loading pickle data."""
        data = {"complex": [1, 2, {"nested": "value"}]}
        file_path = tmp_path / "test.pkl"

        # Create file
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

        # Load and verify
        loaded = PersistenceManager.load_pickle(file_path)
        assert loaded == data

    def test_save_pickle_creates_parent_dirs(self, tmp_path):
        """Test pickle save creates parent directories."""
        file_path = tmp_path / "nested" / "dir" / "test.pkl"
        data = {"key": "value"}

        PersistenceManager.save_pickle(file_path, data)

        assert file_path.exists()
        assert file_path.parent.exists()

    def test_pickle_handles_complex_types(self, tmp_path):
        """Test pickle can handle complex Python types."""
        import numpy as np

        data = {
            "array": np.array([1, 2, 3]),
            "tuple": (1, 2, 3),
            "set": {1, 2, 3},
        }
        file_path = tmp_path / "test.pkl"

        PersistenceManager.save_pickle(file_path, data)
        loaded = PersistenceManager.load_pickle(file_path)

        assert np.array_equal(loaded["array"], data["array"])
        assert loaded["tuple"] == data["tuple"]
        assert loaded["set"] == data["set"]


class TestUpdateDocIdIndices:
    """Test update_doc_id_indices helper."""

    def test_simple_merge(self):
        """Test merging doc_id_to_indices with offset."""
        target = {"doc1": [0, 1, 2]}
        source = {"doc2": [0, 1]}
        offset = 3

        PersistenceManager.update_doc_id_indices(target, source, offset)

        assert target["doc1"] == [0, 1, 2]  # Unchanged
        assert target["doc2"] == [3, 4]  # Offset applied

    def test_merge_with_existing_doc(self):
        """Test merging when document already exists in target."""
        target = {"doc1": [0, 1]}
        source = {"doc1": [0, 1], "doc2": [0]}
        offset = 2

        PersistenceManager.update_doc_id_indices(target, source, offset)

        assert target["doc1"] == [0, 1, 2, 3]  # Extended
        assert target["doc2"] == [2]  # New

    def test_merge_empty_source(self):
        """Test merging empty source does nothing."""
        target = {"doc1": [0, 1]}
        source = {}
        offset = 10

        PersistenceManager.update_doc_id_indices(target, source, offset)

        assert target == {"doc1": [0, 1]}  # Unchanged

    def test_merge_empty_target(self):
        """Test merging into empty target."""
        target = {}
        source = {"doc1": [0, 1], "doc2": [2, 3]}
        offset = 5

        PersistenceManager.update_doc_id_indices(target, source, offset)

        assert target["doc1"] == [5, 6]
        assert target["doc2"] == [7, 8]

    def test_offset_zero(self):
        """Test merge with zero offset."""
        target = {"doc1": [0]}
        source = {"doc2": [0, 1]}
        offset = 0

        PersistenceManager.update_doc_id_indices(target, source, offset)

        assert target["doc2"] == [0, 1]  # No offset

    def test_large_offset(self):
        """Test merge with large offset."""
        target = {}
        source = {"doc1": [0, 1, 2]}
        offset = 1000

        PersistenceManager.update_doc_id_indices(target, source, offset)

        assert target["doc1"] == [1000, 1001, 1002]


class TestVectorStoreLoader:
    """Test VectorStoreLoader format detection."""

    def test_detect_new_format(self, tmp_path):
        """Test detection of new format (JSON + pickle)."""
        # Create new format files
        (tmp_path / "faiss_metadata.json").touch()
        (tmp_path / "faiss_arrays.pkl").touch()

        result = VectorStoreLoader.detect_format(tmp_path)
        assert result == "new"

    def test_detect_old_format(self, tmp_path):
        """Test detection of old format (pickle only)."""
        # Create old format file
        (tmp_path / "metadata.pkl").touch()

        result = VectorStoreLoader.detect_format(tmp_path)
        assert result == "old"

    def test_prefer_new_format_when_both_exist(self, tmp_path):
        """Test new format preferred when both exist."""
        # Create both formats
        (tmp_path / "faiss_metadata.json").touch()
        (tmp_path / "faiss_arrays.pkl").touch()
        (tmp_path / "metadata.pkl").touch()

        result = VectorStoreLoader.detect_format(tmp_path)
        assert result == "new"

    def test_missing_files_raises_error(self, tmp_path):
        """Test error when no valid format found."""
        with pytest.raises(FileNotFoundError):
            VectorStoreLoader.detect_format(tmp_path)

    def test_detect_format_with_path_string(self, tmp_path):
        """Test detection works with string path."""
        (tmp_path / "metadata.pkl").touch()

        result = VectorStoreLoader.detect_format(str(tmp_path))
        assert result == "old"


# Integration tests
class TestPersistenceIntegration:
    """Integration tests for persistence utilities."""

    def test_hybrid_serialization_workflow(self, tmp_path):
        """Test complete hybrid serialization workflow."""
        # Simulate FAISS vector store save/load

        # Config (JSON)
        config = {
            "dimensions": 3072,
            "layer1_count": 1,
            "layer2_count": 10,
            "layer3_count": 100,
        }

        # Arrays (pickle)
        arrays = {
            "metadata_layer1": [{"doc_id": "1", "content": "test"}],
            "metadata_layer2": [{"doc_id": "1", "section": "A"}],
            "doc_id_to_indices": {"1": [0, 1, 2]},
        }

        # Save
        PersistenceManager.save_json(tmp_path / "config.json", config)
        PersistenceManager.save_pickle(tmp_path / "arrays.pkl", arrays)

        # Load
        loaded_config = PersistenceManager.load_json(tmp_path / "config.json")
        loaded_arrays = PersistenceManager.load_pickle(tmp_path / "arrays.pkl")

        # Verify
        assert loaded_config == config
        assert loaded_arrays == arrays

    def test_merge_simulation(self):
        """Test realistic merge scenario."""
        # Store 1: 3 documents
        store1_indices = {
            1: {"doc1": [0, 1], "doc2": [2, 3], "doc3": [4]},
        }

        # Store 2: 2 documents (to be merged into store1)
        store2_indices = {
            1: {"doc4": [0, 1], "doc5": [2]},
        }

        # Merge layer 1 with offset 5 (store1 has 5 chunks)
        offset = 5
        PersistenceManager.update_doc_id_indices(store1_indices[1], store2_indices[1], offset)

        # Verify merge
        assert store1_indices[1]["doc1"] == [0, 1]  # Unchanged
        assert store1_indices[1]["doc2"] == [2, 3]  # Unchanged
        assert store1_indices[1]["doc3"] == [4]  # Unchanged
        assert store1_indices[1]["doc4"] == [5, 6]  # Offset applied
        assert store1_indices[1]["doc5"] == [7]  # Offset applied

    def test_backward_compatibility_load(self, tmp_path):
        """Test loading old format still works."""
        # Create old format
        old_data = {"dimensions": 1024, "metadata": ["chunk1", "chunk2"]}
        with open(tmp_path / "metadata.pkl", "wb") as f:
            pickle.dump(old_data, f)

        # Detect format
        format_type = VectorStoreLoader.detect_format(tmp_path)
        assert format_type == "old"

        # Load
        loaded = PersistenceManager.load_pickle(tmp_path / "metadata.pkl")
        assert loaded == old_data
