import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from prepare_pi05_dataset import merge_table
from pi05_quantile_stats import compute


def test_merge_keeps_action_and_updates_arrow_metadata(tmp_path):
    state = np.arange(16, dtype=np.float32).reshape(2, 8)
    env = np.zeros((2, 14), dtype=np.float32)
    metadata = {b"huggingface": json.dumps({"info": {"features": {
        "observation.state": {"length": 8},
        "observation.environment_state": {"length": 14}}}}).encode()}
    table = pa.table({"observation.state": state.tolist(),
                      "observation.environment_state": env.tolist(),
                      "action": state.tolist()}).replace_schema_metadata(metadata)
    result = merge_table(table)
    assert "observation.environment_state" not in result.column_names
    assert result["action"].equals(table["action"])
    assert json.loads(result.schema.metadata[b"huggingface"])["info"]["features"]["observation.state"]["length"] == 22
    data = tmp_path / "data/chunk-000"
    data.mkdir(parents=True)
    pq.write_table(result, data / "episode_000000.parquet")
    stats = compute(tmp_path)
    assert stats["observation.state"]["dims"] == 22
    low = np.asarray(stats["observation.state"]["q01"])
    high = np.asarray(stats["observation.state"]["q99"])
    assert np.all(high > low)
    assert stats["observation.state"]["epsilon_widened_dimensions"] == list(range(8, 22))
