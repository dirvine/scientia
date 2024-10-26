#!/usr/bin/env python3
import click
import subprocess
import sys
import os
from pathlib import Path
import streamlit.web.cli as stcli
from scientia.core.knowledge_system import ScientiaCore  # Changed from relative to absolute import
import asyncio

@click.group()
def cli():
    """Scientia AI - Knowledge Management System"""
    pass

@cli.command()
def run():
    """Start the Scientia web interface"""
    # Get the path to main.py
    main_path = Path(__file__).parent / "main.py"
    sys.argv = ["streamlit", "run", str(main_path)]
    stcli.main()

@cli.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True))
def add(files):
    """Add documents to the knowledge base"""
    scientia = ScientiaCore()
    
    for file_path in files:
        click.echo(f"Processing {file_path}...")
        with open(file_path, 'rb') as f:
            try:
                knowledge_packets = asyncio.run(
                    scientia.process_document(f, Path(file_path).name)
                )
                for packet in knowledge_packets:
                    asyncio.run(scientia.add_to_knowledge_base(packet))
                click.echo(f"✓ Successfully added {file_path}")
            except Exception as e:
                click.echo(f"✗ Error processing {file_path}: {str(e)}", err=True)

def main():
    cli()

if __name__ == "__main__":
    main()
