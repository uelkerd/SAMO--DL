#!/usr/bin/env python3
"""
Surgical Breakdown Executor

This script helps execute the surgical breakdown of PR #145 into 15 micro-PRs.
It provides automation for creating branches, tracking progress, and ensuring
compliance with PR rules.

Usage:
    python scripts/surgical_breakdown_executor.py --help
    python scripts/surgical_breakdown_executor.py create-pr --pr-number 2
    python scripts/surgical_breakdown_executor.py status
    python scripts/surgical_breakdown_executor.py validate-pr --branch feat/dl-add-t5-summarization-model
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class SurgicalBreakdownExecutor:
    """Executes the surgical breakdown plan for PR #145."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.master_plan_file = self.project_root / "SURGICAL_BREAKDOWN_MASTER_PLAN.md"
        self.progress_file = self.project_root / ".surgical_breakdown_progress.json"
        self.load_progress()
    
    def load_progress(self):
        """Load progress tracking data."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                "created_at": datetime.now().isoformat(),
                "prs": {},
                "current_phase": 1,
                "overall_progress": 0
            }
    
    def save_progress(self):
        """Save progress tracking data."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def create_pr(self, pr_number: int, description: str, files: List[str], 
                  estimated_lines: int, phase: int) -> bool:
        """Create a new micro-PR."""
        branch_name = f"feat/dl-pr-{pr_number:02d}"
        
        print(f"🏗️  Creating PR-{pr_number}: {description}")
        print(f"   Branch: {branch_name}")
        print(f"   Files: {len(files)}")
        print(f"   Estimated lines: {estimated_lines}")
        
        try:
            # Create branch
            subprocess.run(["git", "checkout", "-b", branch_name], check=True)
            
            # Add files
            for file_path in files:
                if Path(file_path).exists():
                    subprocess.run(["git", "add", file_path], check=True)
                else:
                    print(f"   ⚠️  Warning: File {file_path} not found")
            
            # Commit
            commit_message = f"feat: {description.lower()}\n\n- PR-{pr_number} of surgical breakdown\n- Phase {phase}\n- Files: {len(files)}\n- Estimated lines: {estimated_lines}\n\nThis is a focused micro-PR that addresses exactly one concern\nas part of the surgical breakdown of PR #145."
            
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
            
            # Update progress
            self.progress["prs"][str(pr_number)] = {
                "branch": branch_name,
                "description": description,
                "files": files,
                "estimated_lines": estimated_lines,
                "phase": phase,
                "status": "created",
                "created_at": datetime.now().isoformat()
            }
            
            self.save_progress()
            print(f"   ✅ PR-{pr_number} created successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error creating PR-{pr_number}: {e}")
            return False
    
    def validate_pr(self, branch_name: str) -> Dict[str, any]:
        """Validate a PR against our rules."""
        print(f"🔍 Validating PR: {branch_name}")
        
        try:
            # Check file count
            result = subprocess.run(
                ["git", "diff", "--name-only", "main", branch_name],
                capture_output=True, text=True, check=True
            )
            changed_files = [f for f in result.stdout.strip().split('\n') if f]
            file_count = len(changed_files)
            
            # Check line count
            result = subprocess.run(
                ["git", "diff", "--numstat", "main", branch_name],
                capture_output=True, text=True, check=True
            )
            lines_changed = sum(int(line.split('\t')[0]) for line in result.stdout.strip().split('\n') if line)
            
            # Check commit count
            result = subprocess.run(
                ["git", "rev-list", "--count", f"main..{branch_name}"],
                capture_output=True, text=True, check=True
            )
            commit_count = int(result.stdout.strip())
            
            # Validation results
            validation = {
                "file_count": file_count,
                "lines_changed": lines_changed,
                "commit_count": commit_count,
                "file_count_ok": file_count <= 25,
                "lines_ok": lines_changed <= 500,
                "commits_ok": commit_count <= 5,
                "overall_ok": file_count <= 25 and lines_changed <= 500 and commit_count <= 5
            }
            
            print(f"   📊 Files changed: {file_count}/25 {'✅' if validation['file_count_ok'] else '❌'}")
            print(f"   📊 Lines changed: {lines_changed}/500 {'✅' if validation['lines_ok'] else '❌'}")
            print(f"   📊 Commits: {commit_count}/5 {'✅' if validation['commits_ok'] else '❌'}")
            
            if validation['overall_ok']:
                print(f"   ✅ PR validation passed")
            else:
                print(f"   ❌ PR validation failed")
            
            return validation
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error validating PR: {e}")
            return {"error": str(e)}
    
    def status(self):
        """Show current status of all PRs."""
        print("📊 SURGICAL BREAKDOWN STATUS")
        print("=" * 50)
        
        total_prs = 15
        completed = sum(1 for pr in self.progress["prs"].values() if pr["status"] == "completed")
        in_progress = sum(1 for pr in self.progress["prs"].values() if pr["status"] == "in_progress")
        created = sum(1 for pr in self.progress["prs"].values() if pr["status"] == "created")
        
        print(f"Overall Progress: {completed}/{total_prs} PRs completed ({completed/total_prs*100:.1f}%)")
        print(f"Status: {completed} completed, {in_progress} in progress, {created} created")
        print()
        
        for pr_num in range(1, total_prs + 1):
            pr_key = str(pr_num)
            if pr_key in self.progress["prs"]:
                pr = self.progress["prs"][pr_key]
                status_icon = {"completed": "✅", "in_progress": "🚧", "created": "📝"}.get(pr["status"], "⏳")
                print(f"PR-{pr_num:02d}: {status_icon} {pr['description']}")
                print(f"         Branch: {pr['branch']}")
                print(f"         Files: {len(pr['files'])}")
                print(f"         Phase: {pr['phase']}")
                print()
    
    def next_pr(self):
        """Show what the next PR should be."""
        print("🎯 NEXT PR RECOMMENDATION")
        print("=" * 30)
        
        # Find next uncreated PR
        for pr_num in range(1, 16):
            pr_key = str(pr_num)
            if pr_key not in self.progress["prs"]:
                pr_info = self.get_pr_info(pr_num)
                print(f"Next: PR-{pr_num:02d}")
                print(f"Description: {pr_info['description']}")
                print(f"Phase: {pr_info['phase']}")
                print(f"Files: {pr_info['files']}")
                print(f"Estimated lines: {pr_info['estimated_lines']}")
                break
        else:
            print("All PRs have been created!")
    
    def get_pr_info(self, pr_number: int) -> Dict:
        """Get information about a specific PR."""
        pr_info = {
            1: {
                "description": "Add T5 summarization model for journal entries",
                "files": ["src/models/summarization/samo_t5_summarizer.py", "configs/samo_t5_config.yaml", "test_samo_t5_standalone.py"],
                "estimated_lines": 200,
                "phase": 1
            },
            2: {
                "description": "Add Whisper transcription model for voice processing",
                "files": ["src/models/voice_processing/whisper_transcriber.py", "src/models/voice_processing/audio_preprocessor.py"],
                "estimated_lines": 300,
                "phase": 1
            },
            3: {
                "description": "Enhance emotion detection model",
                "files": ["src/models/emotion_detection/bert_classifier.py", "src/models/emotion_detection/labels.py"],
                "estimated_lines": 150,
                "phase": 1
            },
            # Add more PRs as needed
        }
        
        return pr_info.get(pr_number, {
            "description": f"PR-{pr_number} description",
            "files": [],
            "estimated_lines": 100,
            "phase": 1
        })

def main():
    parser = argparse.ArgumentParser(description="Surgical Breakdown Executor")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create PR command
    create_parser = subparsers.add_parser("create-pr", help="Create a new micro-PR")
    create_parser.add_argument("--pr-number", type=int, required=True, help="PR number (1-15)")
    create_parser.add_argument("--description", type=str, help="PR description")
    create_parser.add_argument("--files", nargs="+", help="Files to include")
    create_parser.add_argument("--lines", type=int, help="Estimated lines")
    create_parser.add_argument("--phase", type=int, help="Phase number")
    
    # Validate PR command
    validate_parser = subparsers.add_parser("validate-pr", help="Validate a PR")
    validate_parser.add_argument("--branch", type=str, required=True, help="Branch name to validate")
    
    # Status command
    subparsers.add_parser("status", help="Show current status")
    
    # Next PR command
    subparsers.add_parser("next-pr", help="Show next PR recommendation")
    
    args = parser.parse_args()
    
    executor = SurgicalBreakdownExecutor()
    
    if args.command == "create-pr":
        pr_info = executor.get_pr_info(args.pr_number)
        description = args.description or pr_info["description"]
        files = args.files or pr_info["files"]
        lines = args.lines or pr_info["estimated_lines"]
        phase = args.phase or pr_info["phase"]
        
        executor.create_pr(args.pr_number, description, files, lines, phase)
    
    elif args.command == "validate-pr":
        executor.validate_pr(args.branch)
    
    elif args.command == "status":
        executor.status()
    
    elif args.command == "next-pr":
        executor.next_pr()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
