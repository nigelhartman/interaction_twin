#!/usr/bin/env python3
"""Versioned admin stage context wrapper; original guarded launcher executes."""
from pathlib import Path
import runpy,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    import admin_memory_context
    from admin_quality_learning_v1 import contextual_client,load_catalog
    load_catalog(root=ROOT)
    admin_memory_context.contextual_client=contextual_client
    runpy.run_path(str(ROOT/'scripts/run_admin_stage.py'),run_name='__main__')

if __name__=='__main__':main()
