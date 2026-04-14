#!/usr/bin/env python
"""Debug script to run the trainer with error handling."""
import sys
import traceback

sys.path.insert(0, '.')

try:
    from train.trainer import main
    main([
        '--model-config', 'configs/model/1b.yaml',
        '--schedule-config', 'configs/train/schedule.yaml',
        '--train-shards', 'data/processed/shard-00000.parquet',
        '--validation-shards', 'data/processed/shard-00001.parquet',
        '--output-dir', 'runs/sage-1b-cpu',
        '--steps', '100',
        '--disable-wandb'
    ])
except SystemExit as e:
    print(f"SystemExit: {e.code}")
    sys.exit(e.code)
except Exception as e:
    print("EXCEPTION OCCURRED:")
    traceback.print_exc()
    print(f"ERROR: {e}")
    sys.exit(1)
