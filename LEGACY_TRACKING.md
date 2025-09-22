# Legacy Code Migration Tracker

## 🏰 FORTRESS STATUS
- **Protected Zone**: `src/quality_enforced/` (0 files, ready for new code)
- **Quarantined Legacy**: 636 files (violations unknown until scan)
- **Migration Queue**: Empty (populate as needed)

## 🔴 QUARANTINED FILES (Auto-populated)
<!-- Auto-updated by scripts/update-legacy-tracking.py via CI pipeline on main branch pushes -->
- [ ] src/models/old_model.py (5 violations) #legacy-quarantined
- [ ] scripts/test_script.py (2 violations) #legacy-quarantined
## 🟡 MIGRATION CANDIDATES
Files ready for migration to fortress (manually curated)

## 🟢 QUALITY ENFORCED
- [x] src/quality_enforced/ (protected zone)

## 📋 MIGRATION RULES
1. **NEVER modify quarantined files** - Pre-commit will block
2. **New features ONLY in src/quality_enforced/**
3. **Migration = separate micro-PR** (max 1 file at a time)
4. **CI pipeline updates this file** on pushes to main branch
