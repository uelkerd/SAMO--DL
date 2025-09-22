# Legacy Code Migration Tracker

## 🏰 FORTRESS STATUS
- **Protected Zone**: `src/quality_enforced/` (0 files, ready for new code)
- **Quarantined Legacy**: 636 files (violations unknown until scan)
- **Migration Queue**: Empty (populate as needed)

## 🔴 QUARANTINED FILES (Auto-populated)
<!-- This section will be auto-updated by scripts/update-legacy-tracking.py -->
*Run `python scripts/update-legacy-tracking.py` to populate this section*

## 🟡 MIGRATION CANDIDATES
*Files ready for migration to fortress (manually curated)*

## 🟢 QUALITY ENFORCED
- [x] src/quality_enforced/ (protected zone)

## 📋 MIGRATION RULES
1. **NEVER modify quarantined files** - Pre-commit will block
2. **New features ONLY in src/quality_enforced/**
3. **Migration = separate micro-PR** (max 1 file at a time)
4. **Auto-scan updates this file** on every commit