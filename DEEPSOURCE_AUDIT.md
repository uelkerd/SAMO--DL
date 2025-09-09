# DEEPSOURCE AUDIT REPORT (Branch-wide Total Issues: 2013 (1058 Critical, 358 Major, 597 Minor))

This report catalogs ALL issues identified through CLI analysis. Total occurrences: 2013, unique issues: 77.

---

## Issues for deployment/api_server.py

### Critical Issues (1 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 4 instances. **[RESOLVED]**
  - Affected lines: 33, 42, 62, 82
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances. **[RESOLVED]**
  - Affected lines: 93
- **Rule: FLK-E501** (Line too long) - Total count: 1 instances.
  - Affected lines: 97

### Major Issues (2 total)
- **Rule: FLK-W293** (Blank line contains whitespace) - Total count: 8 instances.
  - Affected lines: 47, 51, 54, 57, 71, 74, 77, 87
- **Rule: PYL-W1203** (Formatted string passed to logging module) - Total count: 3 instances.
  - Affected lines: 30, 59, 79

### Minor Issues (1 total)
- **Rule: BAN-B104** (Audit: Binding to all interfaces detected with hardcoded values) - Total count: 1 instances.
  - Affected lines: 105

---

## Issues for deployment/cloud-run/config.py

### Critical Issues (0 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances. **[RESOLVED]**
  - Affected lines: 215

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: FLK-D204** (1 blank line required after class docstring) - Total count: 1 instances.
  - Affected lines: 12

---

## Issues for deployment/cloud-run/debug_api_import.py

### Critical Issues (1 total)
- **Rule: FLK-E301** (Expected 1 blank line) - Total count: 1 instances.
  - Affected lines: 54

### Major Issues (1 total)
- **Rule: PYL-W0613** (Function contains unused argument) - Total count: 1 instances.
  - Affected lines: 55

### Minor Issues (1 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2

---

## Issues for deployment/cloud-run/debug_errorhandler.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 70

### Minor Issues (2 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: PTC-W0034** (Unnecessary use of `getattr`) - Total count: 1 instances.
  - Affected lines: 42

---

## Issues for deployment/cloud-run/debug_errorhandler_detailed.py

### Critical Issues (1 total)
- **Rule: PYL-E0602** (Undefined name detected) - Total count: 1 instances.
  - Affected lines: 75

### Major Issues (1 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 79

### Minor Issues (3 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: PYL-R1722** (Use of `exit()` or `quit()` detected) - Total count: 2 instances.
  - Affected lines: 17, 25
- **Rule: PTC-W0034** (Unnecessary use of `getattr`) - Total count: 1 instances.
  - Affected lines: 34

---

## Issues for deployment/cloud-run/docs_blueprint.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 27

### Minor Issues (0 total)
- None identified.

---

## Issues for deployment/cloud-run/health_monitor.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 238

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: FLK-D204** (1 blank line required after class docstring) - Total count: 1 instances.
  - Affected lines: 22

---

## Issues for deployment/cloud-run/minimal_api_server.py

### Critical Issues (1 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 1 instances.
  - Affected lines: 31

### Major Issues (1 total)
- **Rule: PYL-W0404** (Multiple imports for an import name detected) - Total count: 1 instances.
  - Affected lines: 11

### Minor Issues (2 total)
- **Rule: PYL-C0412** (Imports from same package are not grouped) - Total count: 1 instances.
  - Affected lines: 11
- **Rule: BAN-B104** (Audit: Binding to all interfaces detected with hardcoded values) - Total count: 1 instances.
  - Affected lines: 158

---

## Issues for deployment/cloud-run/minimal_test.py

### Critical Issues (1 total)
- **Rule: FLK-E301** (Expected 1 blank line) - Total count: 1 instances.
  - Affected lines: 62

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 72
- **Rule: PYL-W0613** (Function contains unused argument) - Total count: 1 instances.
  - Affected lines: 63

### Minor Issues (2 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: PYL-R1722** (Use of `exit()` or `quit()` detected) - Total count: 6 instances.
  - Affected lines: 18, 26, 39, 48, 58, 70

---

## Issues for deployment/cloud-run/model_utils.py

### Critical Issues (1 total)
- **Rule: FLK-E501** (Line too long) - Total count: 1 instances.
  - Affected lines: 190

### Major Issues (1 total)
- **Rule: PYL-W0603** (`global` statement detected) - Total count: 1 instances.
  - Affected lines: 118

### Minor Issues (0 total)
- None identified.

---

## Issues for deployment/cloud-run/onnx_api_server.py

### Critical Issues (1 total)
- **Rule: FLK-E301** (Expected 1 blank line) - Total count: 1 instances.
  - Affected lines: 337

### Major Issues (1 total)
- **Rule: PYL-W0603** (`global` statement detected) - Total count: 1 instances.
  - Affected lines: 221

### Minor Issues (1 total)
- **Rule: PY-D0002** (Missing class docstring) - Total count: 1 instances.
  - Affected lines: 333

---

## Issues for deployment/cloud-run/rate_limiter.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0621** (Re-defined variable from outer scope) - Total count: 1 instances.
  - Affected lines: 34

### Minor Issues (1 total)
- **Rule: PY-D0002** (Missing class docstring) - Total count: 1 instances.
  - Affected lines: 10

---

## Issues for deployment/cloud-run/robust_predict.py

### Critical Issues (2 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 248
- **Rule: FLK-E128** (Continuation line under-indented for visual indent) - Total count: 1 instances.
  - Affected lines: 284

### Major Issues (3 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 304
- **Rule: PYL-W0602** (Global variable is declared but not used) - Total count: 2 instances.
  - Affected lines: 43, 89
- **Rule: PYL-W0621** (Re-defined variable from outer scope) - Total count: 1 instances.
  - Affected lines: 277

### Minor Issues (1 total)
- **Rule: PY-D0002** (Missing class docstring) - Total count: 1 instances.
  - Affected lines: 276

---

## Issues for deployment/cloud-run/secure_api_server.py

### Critical Issues (2 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 3 instances.
  - Affected lines: 59, 476, 502
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 253

### Major Issues (3 total)
- **Rule: PY-W2000** (Imported name is not used anywhere in the module) - Total count: 1 instances.
  - Affected lines: 23
- **Rule: FLK-W505** (Doc line too long) - Total count: 2 instances.
  - Affected lines: 5, 449
- **Rule: PYL-W0613** (Function contains unused argument) - Total count: 3 instances.
  - Affected lines: 450, 460, 465

### Minor Issues (3 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 265
- **Rule: PY-D0002** (Missing class docstring) - Total count: 6 instances.
  - Affected lines: 254, 283, 332, 390, 409, 427
- **Rule: BAN-B104** (Audit: Binding to all interfaces detected with hardcoded values) - Total count: 1 instances.
  - Affected lines: 505

---

## Issues for deployment/cloud-run/security_headers.py

### Critical Issues (2 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 1 instances.
  - Affected lines: 7
- **Rule: FLK-E501** (Line too long) - Total count: 3 instances.
  - Affected lines: 36, 46, 47

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 1 instances.
  - Affected lines: 11

---

## Issues for deployment/cloud-run/test_direct_errorhandler.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (2 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: PYL-R1722** (Use of `exit()` or `quit()` detected) - Total count: 2 instances.
  - Affected lines: 17, 25

---

## Issues for deployment/cloud-run/test_docs_error.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2

---

## Issues for deployment/cloud-run/test_minimal_import.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (2 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: PYL-R1722** (Use of `exit()` or `quit()` detected) - Total count: 5 instances.
  - Affected lines: 18, 26, 34, 44, 53

---

## Issues for deployment/cloud-run/test_minimal_swagger.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (4 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: PYL-R0201** (Consider decorating method with `@staticmethod`) - Total count: 1 instances.
  - Affected lines: 34
- **Rule: PY-D0002** (Missing class docstring) - Total count: 1 instances.
  - Affected lines: 33
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 2 instances.
  - Affected lines: 15, 34

---

## Issues for deployment/cloud-run/test_routing_debug.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (2 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: PY-D0002** (Missing class docstring) - Total count: 1 instances.
  - Affected lines: 36

---

## Issues for deployment/cloud-run/test_routing_fixed.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2

---

## Issues for deployment/cloud-run/test_routing_minimal.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (4 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: PYL-R0201** (Consider decorating method with `@staticmethod`) - Total count: 1 instances.
  - Affected lines: 29
- **Rule: PY-D0002** (Missing class docstring) - Total count: 1 instances.
  - Affected lines: 28
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 4 instances.
  - Affected lines: 29, 34, 39, 44

---

## Issues for deployment/cloud-run/test_server_start.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2

---

## Issues for deployment/cloud-run/test_swagger_debug.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (4 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: PYL-R0201** (Consider decorating method with `@staticmethod`) - Total count: 1 instances.
  - Affected lines: 29
- **Rule: PY-D0002** (Missing class docstring) - Total count: 1 instances.
  - Affected lines: 28
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 2 instances.
  - Affected lines: 29, 34

---

## Issues for deployment/cloud-run/test_swagger_debug_detailed.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2

---

## Issues for deployment/cloud-run/test_swagger_no_model.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (4 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: PYL-R0201** (Consider decorating method with `@staticmethod`) - Total count: 1 instances.
  - Affected lines: 41
- **Rule: PY-D0002** (Missing class docstring) - Total count: 1 instances.
  - Affected lines: 40
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 2 instances.
  - Affected lines: 22, 41

---

## Issues for deployment/gcp/predict.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 2 instances.
  - Affected lines: 91, 146

### Major Issues (0 total)
- None identified.

### Minor Issues (2 total)
- **Rule: PY-D0002** (Missing class docstring) - Total count: 1 instances.
  - Affected lines: 16
- **Rule: BAN-B104** (Audit: Binding to all interfaces detected with hardcoded values) - Total count: 1 instances.
  - Affected lines: 157

---

## Issues for deployment/inference.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 87

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PY-D0002** (Missing class docstring) - Total count: 1 instances.
  - Affected lines: 12

---

## Issues for deployment/local/api_server.py

### Critical Issues (3 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 9 instances.
  - Affected lines: 63, 89, 108, 193, 225, 264, 310, 332, 386
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 2 instances.
  - Affected lines: 190, 393
- **Rule: FLK-E501** (Line too long) - Total count: 18 instances.
  - Affected lines: 72, 80, 106, 116, 125, 138, 162, 170, 210, 281, 306, 316, 320, 321, 322, 347, 358, 408

### Major Issues (2 total)
- **Rule: FLK-W293** (Blank line contains whitespace) - Total count: 43 instances.
  - Affected lines: 69, 74, 82, 85, 94, 103, 113, 117, 124, 127, 131, 135, 139, 142, 149, 152, 160, 163, 181, 183, 198, 213, 216, 218, 230, 233, 238, 244, 247, 250, 252, 269, 272, 277, 283, 289, 292, 298, 337, 374, 377, 379, 411
- **Rule: PYL-W1203** (Formatted string passed to logging module) - Total count: 13 instances.
  - Affected lines: 77, 112, 129, 162, 186, 222, 256, 261, 302, 307, 383, 389, 408

### Minor Issues (3 total)
- **Rule: PTC-W0027** (`f-string` used without any expression) - Total count: 2 instances.
  - Affected lines: 256, 302
- **Rule: PY-D0002** (Missing class docstring) - Total count: 1 instances.
  - Affected lines: 108
- **Rule: BAN-B104** (Audit: Binding to all interfaces detected with hardcoded values) - Total count: 1 instances.
  - Affected lines: 412

---

## Issues for deployment/local/test_api.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 280

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 6 instances.
  - Affected lines: 37, 59, 131, 188, 292, 337

---

## Issues for deployment/secure_api_server.py

### Critical Issues (3 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 15 instances.
  - Affected lines: 105, 129, 348, 426, 436, 568, 608, 674, 899, 925, 942, 959, 1023, 1030, 1036
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 2 instances.
  - Affected lines: 346, 1042
- **Rule: FLK-E501** (Line too long) - Total count: 28 instances.
  - Affected lines: 105, 127, 139, 142, 155, 156, 198, 210, 219, 243, 246, 247, 264, 275, 288, 315, 323, 342, 447, 593, 701, 905, 912, 913, 914, 970, 971, 1069

### Major Issues (7 total)
- **Rule: FLK-W293** (Blank line contains whitespace) - Total count: 49 instances.
  - Affected lines: 110, 121, 124, 136, 149, 161, 164, 167, 169, 173, 178, 268, 272, 286, 289, 292, 299, 310, 313, 316, 339, 573, 596, 599, 601, 613, 623, 628, 637, 644, 653, 657, 665, 667, 679, 689, 694, 703, 710, 723, 730, 741, 933, 950, 964, 1011, 1014, 1016, 1072
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 1073
- **Rule: FLK-W505** (Doc line too long) - Total count: 3 instances.
  - Affected lines: 216, 226, 351
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 4 instances.
  - Affected lines: 660, 661, 726, 1073
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 139
- **Rule: PYL-W1203** (Formatted string passed to logging module) - Total count: 26 instances.
  - Affected lines: 143, 156, 176, 199, 264, 285, 315, 342, 447, 605, 621, 635, 641, 671, 687, 701, 707, 936, 939, 953, 956, 1020, 1026, 1033, 1039, 1069
- **Rule: PYL-W0613** (Function contains unused argument) - Total count: 1 instances.
  - Affected lines: 1031

### Minor Issues (2 total)
- **Rule: PY-D0002** (Missing class docstring) - Total count: 2 instances.
  - Affected lines: 192, 354
- **Rule: BAN-B104** (Audit: Binding to all interfaces detected with hardcoded values) - Total count: 1 instances.
  - Affected lines: 1073

---

## Issues for deployment/test_examples.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0612** (Unused variable found) - Total count: 2 instances.
  - Affected lines: 47, 48

### Minor Issues (1 total)
- **Rule: PYL-R1728** (Redundant list comprehension can be replaced using generator) - Total count: 1 instances.
  - Affected lines: 62

---

## Issues for scripts/ci/api_health_check.py

### Critical Issues (1 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 2 instances.
  - Affected lines: 17, 18

### Major Issues (1 total)
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 57

### Minor Issues (1 total)
- **Rule: PY-D0002** (Missing class docstring) - Total count: 2 instances.
  - Affected lines: 49, 72

---

## Issues for scripts/ci/bert_model_test.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 89

---

## Issues for scripts/ci/model_calibration_test.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 161

---

## Issues for scripts/ci/model_compression_test.py

### Critical Issues (3 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 1 instances.
  - Affected lines: 35
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 13
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 24

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 24

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 167

---

## Issues for scripts/ci/model_monitoring_test.py

### Critical Issues (2 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 25
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 37

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 37

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 235

---

## Issues for scripts/ci/onnx_conversion_test.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 137

---

## Issues for scripts/ci/pre_warm_models.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 46

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/ci/run_full_ci_pipeline.py

### Critical Issues (1 total)
- **Rule: FLK-E128** (Continuation line under-indented for visual indent) - Total count: 1 instances.
  - Affected lines: 379

### Major Issues (6 total)
- **Rule: PYL-W0404** (Multiple imports for an import name detected) - Total count: 5 instances.
  - Affected lines: 231, 232, 261, 268, 269
- **Rule: PYL-W1510** (Subprocess run with ignored non-zero exit) - Total count: 3 instances.
  - Affected lines: 139, 166, 195
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 424
- **Rule: PYL-W0212** (Protected member accessed from outside the class) - Total count: 1 instances.
  - Affected lines: 406
- **Rule: PYL-W0621** (Re-defined variable from outer scope) - Total count: 5 instances.
  - Affected lines: 231, 232, 261, 268, 269
- **Rule: PYL-W0612** (Unused variable found) - Total count: 2 instances.
  - Affected lines: 284, 317

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 4 instances.
  - Affected lines: 146, 173, 202, 291

---

## Issues for scripts/ci/t5_summarization_test.py

### Critical Issues (1 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 1 instances.
  - Affected lines: 27

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 3 instances.
  - Affected lines: 50, 93, 123

---

## Issues for scripts/ci/validation_utils.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: FLK-W391** (Multiple blank lines detected at end of the file) - Total count: 1 instances.
  - Affected lines: 60

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/ci/whisper_transcription_test.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 2 instances.
  - Affected lines: 188, 234

---

## Issues for scripts/deployment/bake_emotion_model.py

### Critical Issues (0 total)
- None identified.

### Major Issues (2 total)
- **Rule: PY-W2000** (Imported name is not used anywhere in the module) - Total count: 1 instances.
  - Affected lines: 3
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 37

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/deployment/complete_project_deployment.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 320

### Major Issues (2 total)
- **Rule: PYL-W1510** (Subprocess run with ignored non-zero exit) - Total count: 3 instances.
  - Affected lines: 58, 86, 258
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 322

### Minor Issues (2 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 2 instances.
  - Affected lines: 62, 90
- **Rule: BAN-B602** (Detected subprocess `popen` call with shell equals `True`) - Total count: 1 instances.
  - Affected lines: 258

---

## Issues for scripts/deployment/convert_model_to_onnx.py

### Critical Issues (1 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 2 instances.
  - Affected lines: 15, 16

### Major Issues (2 total)
- **Rule: FLK-W505** (Doc line too long) - Total count: 1 instances.
  - Affected lines: 73
- **Rule: PYL-W0612** (Unused variable found) - Total count: 2 instances.
  - Affected lines: 120, 160

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/deployment/convert_model_to_onnx_simple.py

### Critical Issues (1 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 2 instances.
  - Affected lines: 15, 16

### Major Issues (1 total)
- **Rule: PYL-W0612** (Unused variable found) - Total count: 2 instances.
  - Affected lines: 111, 151

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/deployment/create_model_deployment_package.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 456

### Major Issues (1 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 457

### Minor Issues (3 total)
- **Rule: PYL-C0201** (Consider iterating dictionary) - Total count: 1 instances.
  - Affected lines: 449
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 11
- **Rule: BAN-B103** (Insecure permissions set on a file) - Total count: 1 instances.
  - Affected lines: 445

---

## Issues for scripts/deployment/deploy_locally.py

### Critical Issues (3 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 1 instances.
  - Affected lines: 16
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 441
- **Rule: FLK-E501** (Line too long) - Total count: 8 instances.
  - Affected lines: 76, 82, 196, 307, 327, 363, 367, 408

### Major Issues (3 total)
- **Rule: PY-W2000** (Imported name is not used anywhere in the module) - Total count: 1 instances.
  - Affected lines: 10
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 443
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 443

### Minor Issues (1 total)
- **Rule: PTC-W0027** (`f-string` used without any expression) - Total count: 1 instances.
  - Affected lines: 416

---

## Issues for scripts/deployment/deploy_to_gcp_vertex_ai.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 485

### Major Issues (2 total)
- **Rule: PYL-W1510** (Subprocess run with ignored non-zero exit) - Total count: 5 instances.
  - Affected lines: 23, 36, 49, 63, 308
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 487

### Minor Issues (2 total)
- **Rule: BAN-B607** (Audit: Starting a process with a partial executable path) - Total count: 13 instances.
  - Affected lines: 23, 36, 49, 63, 308, 332, 339, 345, 357, 375, 391, 401, 411
- **Rule: PTC-W6004** (Audit required: External control of file name or path) - Total count: 1 instances.
  - Affected lines: 443

---

## Issues for scripts/deployment/fix_model_loading_issues.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 306

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: BAN-B103** (Insecure permissions set on a file) - Total count: 1 instances.
  - Affected lines: 187

---

## Issues for scripts/deployment/hf_upload/config_update.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PTC-W6004** (Audit required: External control of file name or path) - Total count: 2 instances.
  - Affected lines: 9, 15

---

## Issues for scripts/deployment/hf_upload/discovery.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (2 total)
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 2 instances.
  - Affected lines: 55, 67
- **Rule: PY-R1000** (Function with cyclomatic complexity higher than threshold) - Total count: 1 instances.
  - Affected lines: 67

---

## Issues for scripts/deployment/hf_upload/prepare.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (3 total)
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 3 instances.
  - Affected lines: 14, 21, 98
- **Rule: PTC-W6004** (Audit required: External control of file name or path) - Total count: 1 instances.
  - Affected lines: 15
- **Rule: PY-R1000** (Function with cyclomatic complexity higher than threshold) - Total count: 2 instances.
  - Affected lines: 21, 98

---

## Issues for scripts/deployment/hf_upload/upload.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (2 total)
- **Rule: BAN-B607** (Audit: Starting a process with a partial executable path) - Total count: 2 instances.
  - Affected lines: 57, 63
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 5 instances.
  - Affected lines: 13, 31, 51, 81, 91

---

## Issues for scripts/deployment/integrate_security_fixes.py

### Critical Issues (2 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 294
- **Rule: FLK-E128** (Continuation line under-indented for visual indent) - Total count: 1 instances.
  - Affected lines: 35

### Major Issues (1 total)
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 217

### Minor Issues (2 total)
- **Rule: PY-D0002** (Missing class docstring) - Total count: 1 instances.
  - Affected lines: 22
- **Rule: BAN-B607** (Audit: Starting a process with a partial executable path) - Total count: 1 instances.
  - Affected lines: 34

---

## Issues for scripts/deployment/save_trained_model_for_deployment.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 197

### Major Issues (1 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 218

### Minor Issues (2 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 2 instances.
  - Affected lines: 15, 152
- **Rule: BAN-B103** (Insecure permissions set on a file) - Total count: 1 instances.
  - Affected lines: 194

---

## Issues for scripts/deployment/security_deployment_fix.py

### Critical Issues (2 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 2 instances.
  - Affected lines: 34, 325
- **Rule: FLK-E128** (Continuation line under-indented for visual indent) - Total count: 1 instances.
  - Affected lines: 28

### Major Issues (0 total)
- None identified.

### Minor Issues (2 total)
- **Rule: PY-D0002** (Missing class docstring) - Total count: 1 instances.
  - Affected lines: 49
- **Rule: BAN-B607** (Audit: Starting a process with a partial executable path) - Total count: 1 instances.
  - Affected lines: 27

---

## Issues for scripts/deployment/vertex_ai_phase4_automation.py

### Critical Issues (2 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 792
- **Rule: FLK-E128** (Continuation line under-indented for visual indent) - Total count: 16 instances.
  - Affected lines: 101, 110, 120, 121, 131, 132, 142, 143, 153, 154, 170, 171, 172, 174, 188, 753

### Major Issues (1 total)
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 173

### Minor Issues (2 total)
- **Rule: FLK-D204** (1 blank line required after class docstring) - Total count: 1 instances.
  - Affected lines: 35
- **Rule: BAN-B607** (Audit: Starting a process with a partial executable path) - Total count: 31 instances.
  - Affected lines: 91, 100, 109, 119, 130, 141, 152, 169, 173, 187, 246, 249, 252, 282, 289, 293, 311, 323, 349, 382, 393, 400, 455, 506, 520, 546, 587, 612, 620, 662, 752

---

## Issues for scripts/ensure_local_emotion_model.py

### Critical Issues (1 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 5 instances.
  - Affected lines: 25, 26, 27, 28, 30

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/legacy/add_comprehensive_features.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 561

### Major Issues (1 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 562

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 13

---

## Issues for scripts/legacy/add_wandb_setup.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 151

### Major Issues (1 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 152

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 13

---

## Issues for scripts/legacy/calibrate_model.py

### Critical Issues (2 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 5
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 24

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 24

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/legacy/comprehensive_model_validation.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 294

### Major Issues (1 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 296

### Minor Issues (4 total)
- **Rule: PTC-W0015** (Unnecessary generator) - Total count: 1 instances.
  - Affected lines: 255
- **Rule: PYL-R1722** (Use of `exit()` or `quit()` detected) - Total count: 1 instances.
  - Affected lines: 296
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 16
- **Rule: PY-R1000** (Function with cyclomatic complexity higher than threshold) - Total count: 1 instances.
  - Affected lines: 15

---

## Issues for scripts/legacy/compress_model.py

### Critical Issues (2 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 25
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 38

### Major Issues (2 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 38
- **Rule: FLK-W505** (Doc line too long) - Total count: 1 instances.
  - Affected lines: 49

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/legacy/convert_to_onnx.py

### Critical Issues (0 total)
- None identified.

### Major Issues (2 total)
- **Rule: FLK-W505** (Doc line too long) - Total count: 2 instances.
  - Affected lines: 12, 13
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 97

### Minor Issues (2 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 209
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 1 instances.
  - Affected lines: 97

---

## Issues for scripts/legacy/create_bulletproof_cell.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 408

### Major Issues (1 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 409

### Minor Issues (2 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 7

---

## Issues for scripts/legacy/create_final_bulletproof_cell.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 444

### Major Issues (1 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 445

### Minor Issues (2 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 7

---

## Issues for scripts/legacy/create_unique_fallback_dataset.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 233

### Major Issues (1 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 237

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 13

---

## Issues for scripts/legacy/deep_model_analysis.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 188

### Major Issues (1 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 190

### Minor Issues (3 total)
- **Rule: PYL-R1722** (Use of `exit()` or `quit()` detected) - Total count: 1 instances.
  - Affected lines: 190
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 13
- **Rule: PY-R1000** (Function with cyclomatic complexity higher than threshold) - Total count: 1 instances.
  - Affected lines: 12

---

## Issues for scripts/legacy/diagnose_f1_issue.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 1 instances.
  - Affected lines: 35

---

## Issues for scripts/legacy/diagnose_model_issue.py

### Critical Issues (2 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 17
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 27

### Major Issues (2 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 27
- **Rule: PYL-W0106** (Expression not assigned) - Total count: 2 instances.
  - Affected lines: 106, 107

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/legacy/evaluate_focal_model.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 259

### Minor Issues (2 total)
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 1 instances.
  - Affected lines: 37
- **Rule: PTC-W0063** (Unguarded next inside generator) - Total count: 2 instances.
  - Affected lines: 99, 172

---

## Issues for scripts/legacy/evaluate_whisper_wer.py

### Critical Issues (1 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 1 instances.
  - Affected lines: 27

### Major Issues (1 total)
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 2 instances.
  - Affected lines: 187, 202

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 142

---

## Issues for scripts/legacy/expand_journal_dataset.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 284

### Major Issues (4 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 285
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 285
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 60
- **Rule: PYL-W0613** (Function contains unused argument) - Total count: 1 instances.
  - Affected lines: 75

### Minor Issues (4 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: PYL-C0201** (Consider iterating dictionary) - Total count: 1 instances.
  - Affected lines: 45
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 76
- **Rule: PTC-W6004** (Audit required: External control of file name or path) - Total count: 1 instances.
  - Affected lines: 17

---

## Issues for scripts/legacy/finalize_emotion_model.py

### Critical Issues (1 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 2 instances.
  - Affected lines: 36, 39

### Major Issues (4 total)
- **Rule: FLK-W505** (Doc line too long) - Total count: 3 instances.
  - Affected lines: 13, 16, 65
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 196
- **Rule: PYL-W0613** (Function contains unused argument) - Total count: 2 instances.
  - Affected lines: 152, 293
- **Rule: PYL-W0511** (Use of `FIXME`/`XXX`/`TODO` encountered) - Total count: 2 instances.
  - Affected lines: 165, 278

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/legacy/fine_tune_emotion_model.py

### Critical Issues (0 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances. **[FALSE POSITIVE - No actual syntax errors found]**
  - Affected lines: 15

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/legacy/improve_model_f1.py

### Critical Issues (1 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 1 instances.
  - Affected lines: 19

### Major Issues (1 total)
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 90

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 43

---

## Issues for scripts/legacy/integrate_cmu_mosei.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 231

### Major Issues (4 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 232
- **Rule: FLK-W505** (Doc line too long) - Total count: 1 instances.
  - Affected lines: 100
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 2 instances.
  - Affected lines: 105, 232
- **Rule: PYL-W0612** (Unused variable found) - Total count: 3 instances.
  - Affected lines: 187, 206, 223

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/legacy/minimal_validation.py

### Critical Issues (0 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances. **[FALSE POSITIVE - No actual syntax errors found]**
  - Affected lines: 4

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/legacy/model_monitoring.py

### Critical Issues (0 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances. **[FALSE POSITIVE - No actual syntax errors found]**
  - Affected lines: 19

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/legacy/model_optimization.py

### Critical Issues (0 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances. **[FALSE POSITIVE - No actual syntax errors found]**
  - Affected lines: 17

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/legacy/optimize_model_performance.py

### Critical Issues (2 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 40
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 56

### Major Issues (2 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 56
- **Rule: FLK-W505** (Doc line too long) - Total count: 1 instances.
  - Affected lines: 58

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 389

---

## Issues for scripts/legacy/optimize_performance.py

### Critical Issues (0 total)
- None identified.

### Major Issues (2 total)
- **Rule: FLK-W505** (Doc line too long) - Total count: 1 instances.
  - Affected lines: 10
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 368

### Minor Issues (1 total)
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 1 instances.
  - Affected lines: 341

---

## Issues for scripts/legacy/reorganize_model_directory.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 280

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 281
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 2 instances.
  - Affected lines: 177, 281

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 18

---

## Issues for scripts/legacy/retrain_with_expanded_dataset.py

### Critical Issues (2 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 294
- **Rule: PYL-E0602** (Undefined name detected) - Total count: 1 instances.
  - Affected lines: 259

### Major Issues (3 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 295
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 295
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 283

### Minor Issues (4 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: PYL-R1721** (Unnecessary use of comprehension) - Total count: 1 instances.
  - Affected lines: 259
- **Rule: PY-D0002** (Missing class docstring) - Total count: 2 instances.
  - Affected lines: 36, 64
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 1 instances.
  - Affected lines: 72

---

## Issues for scripts/legacy/retrain_with_validation.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 399

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 401
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 401

### Minor Issues (2 total)
- **Rule: PYL-R1722** (Use of `exit()` or `quit()` detected) - Total count: 1 instances.
  - Affected lines: 401
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 2 instances.
  - Affected lines: 10, 54

---

## Issues for scripts/legacy/simple_cmu_mosei_download.py

### Critical Issues (2 total)
- **Rule: FLK-E228** (Missing whitespace around modulo operator) - Total count: 1 instances.
  - Affected lines: 93
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 227

### Major Issues (3 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 228
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 2 instances.
  - Affected lines: 106, 228
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 215

### Minor Issues (1 total)
- **Rule: PTC-W6004** (Audit required: External control of file name or path) - Total count: 1 instances.
  - Affected lines: 168

---

## Issues for scripts/legacy/simple_f1_evaluation.py

### Critical Issues (1 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 3 instances.
  - Affected lines: 18, 19, 20

### Major Issues (3 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 189
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 189
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 41

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/legacy/simple_finalize_model.py

### Critical Issues (0 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances. **[FALSE POSITIVE - No actual syntax errors found]**
  - Affected lines: 8

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/legacy/simple_validation.py

### Critical Issues (0 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances. **[FALSE POSITIVE - No actual syntax errors found]**
  - Affected lines: 2

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/legacy/simple_vertex_ai_validation.py

### Critical Issues (0 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances. **[FALSE POSITIVE - No actual syntax errors found]**
  - Affected lines: 5

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/legacy/start_monitoring_dashboard.py

### Critical Issues (0 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances. **[FALSE POSITIVE - No actual syntax errors found]**
  - Affected lines: 5

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/legacy/temperature_scaling.py

### Critical Issues (0 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances. **[FALSE POSITIVE - No actual syntax errors found]**
  - Affected lines: 8

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/legacy/threshold_optimization.py

### Critical Issues (0 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances. **[FALSE POSITIVE - No actual syntax errors found]**
  - Affected lines: 9

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/legacy/trigger_ci.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (2 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 21

---

## Issues for scripts/legacy/update_model_threshold.py

### Critical Issues (2 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 11
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 23

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 23

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/legacy/validate_and_train.py

### Critical Issues (1 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances.
  - Affected lines: 4

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/legacy/validate_current_f1.py

### Critical Issues (2 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 3
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 9

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 9

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/legacy/validate_model_performance.py

### Critical Issues (2 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 316
- **Rule: FLK-E722** (Do not use bare `except`, specify exception instead) - Total count: 1 instances.
  - Affected lines: 295

### Major Issues (3 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 317
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 317
- **Rule: PYL-W0612** (Unused variable found) - Total count: 3 instances.
  - Affected lines: 150, 286, 294

### Minor Issues (3 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 2 instances.
  - Affected lines: 51, 247
- **Rule: PTC-W0063** (Unguarded next inside generator) - Total count: 1 instances.
  - Affected lines: 142
- **Rule: PTC-W6004** (Audit required: External control of file name or path) - Total count: 1 instances.
  - Affected lines: 33

---

## Issues for scripts/legacy/vertex_ai_setup.py

### Critical Issues (1 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances.
  - Affected lines: 12

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/maintenance/auto_fix_code_quality.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 450

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/maintenance/code_quality_report.py

### Critical Issues (3 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 4
- **Rule: FLK-E266** (Too many leading `#` for block comment) - Total count: 3 instances.
  - Affected lines: 5, 6, 7
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 18

### Major Issues (2 total)
- **Rule: PYL-W0404** (Multiple imports for an import name detected) - Total count: 1 instances.
  - Affected lines: 10
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 18

### Minor Issues (1 total)
- **Rule: BAN-B607** (Audit: Starting a process with a partial executable path) - Total count: 1 instances.
  - Affected lines: 27

---

## Issues for scripts/maintenance/emergency_f1_fix.py

### Critical Issues (1 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 2 instances.
  - Affected lines: 31, 32

### Major Issues (3 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 392
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 2 instances.
  - Affected lines: 165, 392
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 313

### Minor Issues (1 total)
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 2 instances.
  - Affected lines: 48, 83

---

## Issues for scripts/maintenance/fix_all_imports_aggressive.py

### Critical Issues (4 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 17
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 153
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 25
- **Rule: PYL-E0602** (Undefined name detected) - Total count: 1 instances.
  - Affected lines: 69

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 25

### Minor Issues (2 total)
- **Rule: PTC-W6004** (Audit required: External control of file name or path) - Total count: 2 instances.
  - Affected lines: 32, 96
- **Rule: PY-R1000** (Function with cyclomatic complexity higher than threshold) - Total count: 1 instances.
  - Affected lines: 30

---

## Issues for scripts/maintenance/fix_ci_issues.py

### Critical Issues (2 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 7
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 19

### Major Issues (2 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 19
- **Rule: PYL-W0613** (Function contains unused argument) - Total count: 1 instances.
  - Affected lines: 23

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 2 instances.
  - Affected lines: 30, 71

---

## Issues for scripts/maintenance/fix_code_quality.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (2 total)
- **Rule: PTC-W0048** (`if` statements can be merged) - Total count: 1 instances.
  - Affected lines: 29
- **Rule: PTC-W0051** (Branches of the `if` statement have similar implementation) - Total count: 1 instances.
  - Affected lines: 92

---

## Issues for scripts/maintenance/fix_import_paths.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 75

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 76
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 3 instances.
  - Affected lines: 33, 35, 76

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 43

---

## Issues for scripts/maintenance/fix_label_mapping.py

### Critical Issues (2 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 3 instances.
  - Affected lines: 25, 26, 27
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 2 instances.
  - Affected lines: 21, 516

### Major Issues (3 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 529
- **Rule: PYL-W0621** (Re-defined variable from outer scope) - Total count: 2 instances.
  - Affected lines: 41, 53
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 529

### Minor Issues (2 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 112

---

## Issues for scripts/maintenance/fix_linting.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (2 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: PTC-W6004** (Audit required: External control of file name or path) - Total count: 1 instances.
  - Affected lines: 17

---

## Issues for scripts/maintenance/fix_linting_issues.py

### Critical Issues (2 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 10
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 18

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 18

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/maintenance/fix_linting_issues_comprehensive.py

### Critical Issues (4 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 18
- **Rule: FLK-E129** (Visually indented line with same indent as next logical line) - Total count: 2 instances.
  - Affected lines: 77, 124
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 48
- **Rule: PYL-E0602** (Undefined name detected) - Total count: 1 instances.
  - Affected lines: 183

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 24

### Minor Issues (4 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 158
- **Rule: PTC-W0048** (`if` statements can be merged) - Total count: 2 instances.
  - Affected lines: 149, 181
- **Rule: PTC-W0051** (Branches of the `if` statement have similar implementation) - Total count: 1 instances.
  - Affected lines: 90
- **Rule: PTC-W6004** (Audit required: External control of file name or path) - Total count: 1 instances.
  - Affected lines: 106

---

## Issues for scripts/maintenance/fix_linting_issues_conservative.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 242

### Minor Issues (1 total)
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 2 instances.
  - Affected lines: 58, 112

---

## Issues for scripts/maintenance/fix_model_architecture_mismatch.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 80

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 81
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 81

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 13

---

## Issues for scripts/maintenance/fix_model_reconfiguration.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 91

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 92
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 92

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 14

---

## Issues for scripts/maintenance/fix_remaining_linting.py

### Critical Issues (3 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 18
- **Rule: FLK-E129** (Visually indented line with same indent as next logical line) - Total count: 1 instances.
  - Affected lines: 116
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 37

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 21

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/maintenance/fix_remaining_py38_types.py

### Critical Issues (1 total)
- **Rule: FLK-E129** (Visually indented line with same indent as next logical line) - Total count: 1 instances.
  - Affected lines: 140

### Major Issues (1 total)
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 263

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/maintenance/fix_threshold_tuning.py

### Critical Issues (2 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 8
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 19

### Major Issues (2 total)
- **Rule: PYL-W0104** (Statement has no effect) - Total count: 1 instances.
  - Affected lines: 70
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 19

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 80

---

## Issues for scripts/maintenance/improve_model_f1_fixed.py

### Critical Issues (6 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 10 instances.
  - Affected lines: 47, 48, 49, 50, 51, 52, 53, 54, 55, 56
- **Rule: PYL-E1121** (Too many positional arguments in function call) - Total count: 2 instances.
  - Affected lines: 159, 209
- **Rule: PYL-E0633** (Attempting to unpack a non-sequence object) - Total count: 1 instances.
  - Affected lines: 296
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 40
- **Rule: PYL-E1123** (Unexpected keyword argument in function call) - Total count: 2 instances.
  - Affected lines: 147, 197
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 61

### Major Issues (5 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 61
- **Rule: PYL-W0621** (Re-defined variable from outer scope) - Total count: 3 instances.
  - Affected lines: 115, 170, 280
- **Rule: FLK-W505** (Doc line too long) - Total count: 1 instances.
  - Affected lines: 68
- **Rule: PYL-W0612** (Unused variable found) - Total count: 2 instances.
  - Affected lines: 189, 296
- **Rule: PYL-W0613** (Function contains unused argument) - Total count: 1 instances.
  - Affected lines: 280

### Minor Issues (2 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 120
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 1 instances.
  - Affected lines: 97

---

## Issues for scripts/maintenance/quick_label_fix.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 69

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 71
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 71

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/maintenance/typehint_codemod.py

### Critical Issues (2 total)
- **Rule: FLK-E301** (Expected 1 blank line) - Total count: 1 instances.
  - Affected lines: 25
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 100

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PTC-W6004** (Audit required: External control of file name or path) - Total count: 2 instances.
  - Affected lines: 259, 290

---

## Issues for scripts/maintenance/vertex_ai_setup_fixed.py

### Critical Issues (1 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances.
  - Affected lines: 8

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/testing/basic_environment_test.py

### Critical Issues (2 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 5
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 2 instances.
  - Affected lines: 11, 18

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 11

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 66

---

## Issues for scripts/testing/check_model_health.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PY-W2000** (Imported name is not used anywhere in the module) - Total count: 1 instances.
  - Affected lines: 8

### Minor Issues (1 total)
- **Rule: PYL-R1722** (Use of `exit()` or `quit()` detected) - Total count: 1 instances.
  - Affected lines: 73

---

## Issues for scripts/testing/config.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: FLK-W505** (Doc line too long) - Total count: 1 instances.
  - Affected lines: 4

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/testing/create_journal_test_dataset.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 308

### Major Issues (3 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 309
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 309
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 217

### Minor Issues (1 total)
- **Rule: PTC-W6004** (Audit required: External control of file name or path) - Total count: 1 instances.
  - Affected lines: 243

---

## Issues for scripts/testing/create_test_dataset.py

### Critical Issues (2 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 10
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 19

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 19

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 24

---

## Issues for scripts/testing/debug_checkpoint.py

### Critical Issues (2 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 10

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 10

### Minor Issues (1 total)
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 1 instances.
  - Affected lines: 14

---

## Issues for scripts/testing/debug_dataset_structure.py

### Critical Issues (1 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 1 instances.
  - Affected lines: 15

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PYL-C0201** (Consider iterating dictionary) - Total count: 1 instances.
  - Affected lines: 35

---

## Issues for scripts/testing/debug_evaluation_step_by_step.py

### Critical Issues (3 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 15
- **Rule: PYL-E1123** (Unexpected keyword argument in function call) - Total count: 1 instances.
  - Affected lines: 41
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 26

### Major Issues (2 total)
- **Rule: PYL-W0104** (Statement has no effect) - Total count: 1 instances.
  - Affected lines: 135
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 26

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 37

---

## Issues for scripts/testing/debug_go_emotions_labels.py

### Critical Issues (3 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 1 instances.
  - Affected lines: 25
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 2 instances.
  - Affected lines: 21, 103
- **Rule: FLK-E722** (Do not use bare `except`, specify exception instead) - Total count: 2 instances.
  - Affected lines: 57, 64

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 104
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 104

### Minor Issues (1 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2

---

## Issues for scripts/testing/debug_label_mismatch.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 214

### Major Issues (5 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 221
- **Rule: PYL-W0631** (Loop variable used outside the loop) - Total count: 1 instances.
  - Affected lines: 112
- **Rule: FLK-W505** (Doc line too long) - Total count: 1 instances.
  - Affected lines: 3
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 221
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 101

### Minor Issues (1 total)
- **Rule: PY-R1000** (Function with cyclomatic complexity higher than threshold) - Total count: 1 instances.
  - Affected lines: 16

---

## Issues for scripts/testing/debug_model_loading.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PY-W2000** (Imported name is not used anywhere in the module) - Total count: 2 instances.
  - Affected lines: 9, 10

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/testing/debug_rate_limiter_test.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 1

### Minor Issues (1 total)
- **Rule: PTC-W0030** (Empty module found) - Total count: 1 instances.
  - Affected lines: 1

---

## Issues for scripts/testing/debug_state_dict.py

### Critical Issues (2 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 10

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 10

### Minor Issues (1 total)
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 1 instances.
  - Affected lines: 14

---

## Issues for scripts/testing/direct_evaluation_test.py

### Critical Issues (3 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 17
- **Rule: PYL-E1123** (Unexpected keyword argument in function call) - Total count: 1 instances.
  - Affected lines: 43
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 28

### Major Issues (2 total)
- **Rule: PYL-W0104** (Statement has no effect) - Total count: 2 instances.
  - Affected lines: 109, 150
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 28

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 39

---

## Issues for scripts/testing/final_temperature_test.py

### Critical Issues (1 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 2 instances.
  - Affected lines: 18, 19

### Major Issues (0 total)
- None identified.

### Minor Issues (2 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: PY-R1000** (Function with cyclomatic complexity higher than threshold) - Total count: 1 instances.
  - Affected lines: 22

---

## Issues for scripts/testing/hf_serverless_smoke.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: FLK-W505** (Doc line too long) - Total count: 1 instances.
  - Affected lines: 77

### Minor Issues (1 total)
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 3 instances.
  - Affected lines: 29, 41, 60

---

## Issues for scripts/testing/local_validation_debug.py

### Critical Issues (1 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances.
  - Affected lines: 22

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/testing/mega_comprehensive_model_test.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 720

### Major Issues (5 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 721
- **Rule: PY-W0069** (Consider removing the commented out code block) - Total count: 1 instances.
  - Affected lines: 18
- **Rule: FLK-W505** (Doc line too long) - Total count: 1 instances.
  - Affected lines: 6
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 721
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 627

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/testing/mega_test_summary.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 147

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 148
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 148

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 10

---

## Issues for scripts/testing/minimal_eval_test.py

### Critical Issues (1 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 7

### Major Issues (2 total)
- **Rule: PYL-W0104** (Statement has no effect) - Total count: 1 instances.
  - Affected lines: 37
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 12

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 21

---

## Issues for scripts/testing/minimal_test.py

### Critical Issues (1 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 1 instances.
  - Affected lines: 17

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/testing/quick_f1_test.py

### Critical Issues (1 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances.
  - Affected lines: 8

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/testing/quick_focal_test.py

### Critical Issues (1 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances.
  - Affected lines: 9

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/testing/quick_temperature_test.py

### Critical Issues (2 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 6
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 18

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 18

### Minor Issues (1 total)
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 1 instances.
  - Affected lines: 24

---

## Issues for scripts/testing/run_api_rate_limiter_tests.py

### Critical Issues (1 total)
- **Rule: FLK-E501** (Line too long) - Total count: 1 instances.
  - Affected lines: 40

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/testing/setup_model_testing.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 162

### Major Issues (3 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 168
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 2 instances.
  - Affected lines: 45, 168
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 117

### Minor Issues (2 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: PTC-W0048** (`if` statements can be merged) - Total count: 1 instances.
  - Affected lines: 120

---

## Issues for scripts/testing/simple_loss_debug.py

### Critical Issues (2 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 14
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 21

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 21

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/testing/simple_model_test.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 130

### Major Issues (4 total)
- **Rule: PYL-W0404** (Multiple imports for an import name detected) - Total count: 2 instances.
  - Affected lines: 68, 76
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 131
- **Rule: PYL-W0621** (Re-defined variable from outer scope) - Total count: 2 instances.
  - Affected lines: 68, 76
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 131

### Minor Issues (1 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2

---

## Issues for scripts/testing/simple_rate_limiter_test.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 1

### Minor Issues (1 total)
- **Rule: PTC-W0030** (Empty module found) - Total count: 1 instances.
  - Affected lines: 1

---

## Issues for scripts/testing/simple_temperature_test.py

### Critical Issues (3 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 1 instances.
  - Affected lines: 17
- **Rule: PYL-E1120** (Missing argument in function call) - Total count: 1 instances.
  - Affected lines: 72
- **Rule: PYL-E1123** (Unexpected keyword argument in function call) - Total count: 1 instances.
  - Affected lines: 72

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/testing/simple_temperature_test_local.py

### Critical Issues (1 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances.
  - Affected lines: 12

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/testing/simple_test.py

### Critical Issues (1 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances.
  - Affected lines: 4

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/testing/simple_threshold_test.py

### Critical Issues (2 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 9
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 16

### Major Issues (2 total)
- **Rule: PYL-W0104** (Statement has no effect) - Total count: 1 instances.
  - Affected lines: 42
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 16

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 21

---

## Issues for scripts/testing/standalone_focal_test.py

### Critical Issues (1 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances.
  - Affected lines: 5

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/testing/test_api_startup.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PTC-W0030** (Empty module found) - Total count: 1 instances.
  - Affected lines: 1

---

## Issues for scripts/testing/test_calibration.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 30

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 115

---

## Issues for scripts/testing/test_calibration_fixed.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 28

### Minor Issues (2 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 208
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 1 instances.
  - Affected lines: 71

---

## Issues for scripts/testing/test_cloud_run_api_endpoints.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0612** (Unused variable found) - Total count: 3 instances.
  - Affected lines: 230, 260, 297

### Minor Issues (1 total)
- **Rule: PY-D0002** (Missing class docstring) - Total count: 1 instances.
  - Affected lines: 21

---

## Issues for scripts/testing/test_comprehensive_model.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (4 total)
- **Rule: PTC-W0027** (`f-string` used without any expression) - Total count: 11 instances.
  - Affected lines: 93, 113, 212, 228, 267, 271, 292, 318, 349, 366, 391
- **Rule: PTC-W0060** (Implicit enumerate calls found) - Total count: 1 instances.
  - Affected lines: 72
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 17
- **Rule: PY-R1000** (Function with cyclomatic complexity higher than threshold) - Total count: 1 instances.
  - Affected lines: 16

---

## Issues for scripts/testing/test_domain_adaptation.py

### Critical Issues (1 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances.
  - Affected lines: 25

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/testing/test_e2e_simple.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PTC-W0030** (Empty module found) - Total count: 1 instances.
  - Affected lines: 1

---

## Issues for scripts/testing/test_emotion_model.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (4 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: PTC-W0027** (`f-string` used without any expression) - Total count: 1 instances.
  - Affected lines: 139
- **Rule: PY-D0002** (Missing class docstring) - Total count: 1 instances.
  - Affected lines: 38
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 1 instances.
  - Affected lines: 46

---

## Issues for scripts/testing/test_final_inference.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (2 total)
- **Rule: PTC-W0027** (`f-string` used without any expression) - Total count: 11 instances.
  - Affected lines: 37, 70, 87, 120, 181, 187, 207, 208, 209, 210, 212
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 2 instances.
  - Affected lines: 13, 140

---

## Issues for scripts/testing/test_fixed_evaluation.py

### Critical Issues (0 total)
- None identified.

### Major Issues (2 total)
- **Rule: PYL-W0104** (Statement has no effect) - Total count: 1 instances.
  - Affected lines: 72
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 19

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 83

---

## Issues for scripts/testing/test_fixed_inference.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (2 total)
- **Rule: PTC-W0027** (`f-string` used without any expression) - Total count: 9 instances.
  - Affected lines: 37, 70, 87, 120, 146, 147, 148, 149, 151
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 13

---

## Issues for scripts/testing/test_loss_scenarios.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 14

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/testing/test_model_status.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PYL-R1722** (Use of `exit()` or `quit()` detected) - Total count: 1 instances.
  - Affected lines: 101

---

## Issues for scripts/testing/test_new_trained_model.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (2 total)
- **Rule: PTC-W0027** (`f-string` used without any expression) - Total count: 8 instances.
  - Affected lines: 44, 55, 108, 129, 139, 140, 141, 142
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 12

---

## Issues for scripts/testing/test_new_trained_model_comprehensive.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (5 total)
- **Rule: PTC-W0027** (`f-string` used without any expression) - Total count: 7 instances.
  - Affected lines: 213, 221, 233, 234, 235, 236, 244
- **Rule: PTC-W0060** (Implicit enumerate calls found) - Total count: 1 instances.
  - Affected lines: 65
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 20
- **Rule: PYL-R1710** (Inconsistent return statements) - Total count: 1 instances.
  - Affected lines: 19
- **Rule: PY-R1000** (Function with cyclomatic complexity higher than threshold) - Total count: 1 instances.
  - Affected lines: 19

---

## Issues for scripts/testing/test_numpy_compatibility.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 46

### Minor Issues (3 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 37
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 1 instances.
  - Affected lines: 27

---

## Issues for scripts/testing/test_phase3_cloud_run_optimization.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 156

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/testing/test_phase3_cloud_run_optimization_fixed.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0612** (Unused variable found) - Total count: 3 instances.
  - Affected lines: 148, 176, 206

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/testing/test_pr4_integration.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W1510** (Subprocess run with ignored non-zero exit) - Total count: 2 instances.
  - Affected lines: 272, 291

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/testing/test_pr5_cicd_integration.py

### Critical Issues (0 total)
- None identified.

### Major Issues (3 total)
- **Rule: PYL-W0404** (Multiple imports for an import name detected) - Total count: 1 instances.
  - Affected lines: 97
- **Rule: PYL-W1510** (Subprocess run with ignored non-zero exit) - Total count: 1 instances.
  - Affected lines: 44
- **Rule: PYL-W0612** (Unused variable found) - Total count: 3 instances.
  - Affected lines: 127, 170, 339

### Minor Issues (2 total)
- **Rule: PTC-W0027** (`f-string` used without any expression) - Total count: 1 instances.
  - Affected lines: 88
- **Rule: PY-R1000** (Function with cyclomatic complexity higher than threshold) - Total count: 2 instances.
  - Affected lines: 95, 250

---

## Issues for scripts/testing/test_rate_limiter_no_threading.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PTC-W0030** (Empty module found) - Total count: 1 instances.
  - Affected lines: 1

---

## Issues for scripts/testing/test_temperature_scaling.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 25

### Minor Issues (2 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 39
- **Rule: PYL-R1710** (Inconsistent return statements) - Total count: 1 instances.
  - Affected lines: 38

---

## Issues for scripts/testing/test_working_inference.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (2 total)
- **Rule: PTC-W0027** (`f-string` used without any expression) - Total count: 7 instances.
  - Affected lines: 51, 72, 94, 134, 156, 157, 159
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 2 instances.
  - Affected lines: 13, 102

---

## Issues for scripts/training/SAMO_Colab_Setup.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 300

### Major Issues (1 total)
- **Rule: PYL-W0106** (Expression not assigned) - Total count: 1 instances.
  - Affected lines: 29

### Minor Issues (1 total)
- **Rule: BAN-B607** (Audit: Starting a process with a partial executable path) - Total count: 3 instances.
  - Affected lines: 41, 55, 70

---

## Issues for scripts/training/add_advanced_features_to_notebook.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 629

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 630
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 630

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 15

---

## Issues for scripts/training/bulletproof_training.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 446

### Major Issues (3 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 449
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 449
- **Rule: PYL-W0612** (Unused variable found) - Total count: 2 instances.
  - Affected lines: 255, 411

### Minor Issues (4 total)
- **Rule: PTC-W0027** (`f-string` used without any expression) - Total count: 2 instances.
  - Affected lines: 152, 156
- **Rule: FLK-D204** (1 blank line required after class docstring) - Total count: 2 instances.
  - Affected lines: 163, 208
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 1 instances.
  - Affected lines: 222
- **Rule: PY-R1000** (Function with cyclomatic complexity higher than threshold) - Total count: 1 instances.
  - Affected lines: 240

---

## Issues for scripts/training/bulletproof_training_cell.py

### Critical Issues (1 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances.
  - Affected lines: 44

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/training/bulletproof_training_cell_fixed.py

### Critical Issues (1 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances.
  - Affected lines: 44

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/training/complete_simple_notebook.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 490

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 491
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 491

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 13

---

## Issues for scripts/training/comprehensive_domain_adaptation_training.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 706

### Major Issues (4 total)
- **Rule: PYL-W1510** (Subprocess run with ignored non-zero exit) - Total count: 5 instances.
  - Affected lines: 109, 116, 130, 141, 264
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 709
- **Rule: FLK-W505** (Doc line too long) - Total count: 1 instances.
  - Affected lines: 8
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 15 instances.
  - Affected lines: 110, 117, 118, 120, 131, 142, 143, 144, 145, 146, 147, 148, 149, 150, 709

### Minor Issues (8 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 3 instances.
  - Affected lines: 265, 418, 513
- **Rule: PTC-W0048** (`if` statements can be merged) - Total count: 1 instances.
  - Affected lines: 279
- **Rule: PYL-R1728** (Redundant list comprehension can be replaced using generator) - Total count: 2 instances.
  - Affected lines: 398, 399
- **Rule: BAN-B602** (Detected subprocess `popen` call with shell equals `True`) - Total count: 1 instances.
  - Affected lines: 264
- **Rule: FLK-D204** (1 blank line required after class docstring) - Total count: 1 instances.
  - Affected lines: 51
- **Rule: BAN-B607** (Audit: Starting a process with a partial executable path) - Total count: 4 instances.
  - Affected lines: 109, 116, 130, 141
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 5 instances.
  - Affected lines: 163, 214, 236, 390, 555
- **Rule: PY-R1000** (Function with cyclomatic complexity higher than threshold) - Total count: 1 instances.
  - Affected lines: 378

---

## Issues for scripts/training/create_bulletproof_colab_notebook.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 716

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 717
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 717

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 13

---

## Issues for scripts/training/create_colab_expanded_training.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 736

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 737
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 737

### Minor Issues (2 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 7

---

## Issues for scripts/training/create_colab_notebook.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 675

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 676
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 676

### Minor Issues (2 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 9

---

## Issues for scripts/training/create_comprehensive_notebook.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 602

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 603
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 603

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 13

---

## Issues for scripts/training/create_corrected_specialized_notebook.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 643

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 645
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 645

### Minor Issues (2 total)
- **Rule: PTC-W0027** (`f-string` used without any expression) - Total count: 13 instances.
  - Affected lines: 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 11

---

## Issues for scripts/training/create_emotion_specialized_notebook.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 501

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 502
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 502

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 12

---

## Issues for scripts/training/create_final_bulletproof_notebook.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 735

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 736
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 736

### Minor Issues (2 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 9

---

## Issues for scripts/training/create_final_colab_notebook.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 484

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 485
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 485

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 12

---

## Issues for scripts/training/create_fixed_bulletproof_notebook.py

### Critical Issues (2 total)
- **Rule: FLK-E131** (Continuation line unaligned for hanging indent) - Total count: 1 instances.
  - Affected lines: 45
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 470

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 471
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 471

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 12

---

## Issues for scripts/training/create_fixed_colab_notebook.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 455

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 456
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 456

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 12

---

## Issues for scripts/training/create_fixed_notebook.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 647

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 649
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 649

### Minor Issues (2 total)
- **Rule: PTC-W0027** (`f-string` used without any expression) - Total count: 13 instances.
  - Affected lines: 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 13

---

## Issues for scripts/training/create_fixed_specialized_training_notebook.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 682

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 683
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 683

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 16

---

## Issues for scripts/training/create_improved_expanded_notebook.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 766

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 767
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 767

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 10

---

## Issues for scripts/training/create_minimal_working_notebook.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 381

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 382
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 382

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 13

---

## Issues for scripts/training/create_model_ensemble_notebook.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 676

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 677
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 677

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 12

---

## Issues for scripts/training/create_simple_ultimate_notebook.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 416

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 417
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 417

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 13

---

## Issues for scripts/training/create_ultimate_bulletproof_notebook.py

### Critical Issues (1 total)
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 419

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 420
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 2 instances.
  - Affected lines: 10, 420

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 21

---

## Issues for scripts/training/debug_colab_compatibility.py

### Critical Issues (2 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 5 instances.
  - Affected lines: 167, 193, 228, 250, 286
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 320

### Major Issues (4 total)
- **Rule: PYL-W1510** (Subprocess run with ignored non-zero exit) - Total count: 1 instances.
  - Affected lines: 22
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 321
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 321
- **Rule: PYL-W0612** (Unused variable found) - Total count: 5 instances.
  - Affected lines: 78, 85, 106, 107, 211

### Minor Issues (3 total)
- **Rule: PTC-W0027** (`f-string` used without any expression) - Total count: 1 instances.
  - Affected lines: 55
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 6 instances.
  - Affected lines: 23, 39, 54, 123, 160, 186
- **Rule: BAN-B602** (Detected subprocess `popen` call with shell equals `True`) - Total count: 1 instances.
  - Affected lines: 22

---

## Issues for scripts/training/debug_training_loss.py

### Critical Issues (2 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 3 instances.
  - Affected lines: 22, 23, 24
- **Rule: PYL-E1123** (Unexpected keyword argument in function call) - Total count: 3 instances.
  - Affected lines: 36, 97, 250

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/training/final_bulletproof_training_cell.py

### Critical Issues (1 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances.
  - Affected lines: 44

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/training/final_combined_training.py

### Critical Issues (2 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 4 instances.
  - Affected lines: 36, 105, 135, 148
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 274

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 275
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 4 instances.
  - Affected lines: 21, 22, 23, 275

### Minor Issues (1 total)
- **Rule: PTC-W0027** (`f-string` used without any expression) - Total count: 1 instances.
  - Affected lines: 271

---

## Issues for scripts/training/final_expanded_training.py

### Critical Issues (3 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 2 instances.
  - Affected lines: 62, 128
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 2 instances.
  - Affected lines: 91, 141
- **Rule: FLK-E128** (Continuation line under-indented for visual indent) - Total count: 1 instances.
  - Affected lines: 184

### Major Issues (4 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 237
- **Rule: PY-W0069** (Consider removing the commented out code block) - Total count: 1 instances.
  - Affected lines: 121
- **Rule: PYL-W0621** (Re-defined variable from outer scope) - Total count: 2 instances.
  - Affected lines: 63, 73
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 7 instances.
  - Affected lines: 10, 19, 20, 21, 95, 183, 237

### Minor Issues (3 total)
- **Rule: PTC-W0027** (`f-string` used without any expression) - Total count: 7 instances.
  - Affected lines: 157, 216, 224, 231, 234, 236, 237
- **Rule: PY-D0002** (Missing class docstring) - Total count: 1 instances.
  - Affected lines: 62
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 1 instances.
  - Affected lines: 128

---

## Issues for scripts/training/fix_imports_in_notebook.py

### Critical Issues (2 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 1 instances.
  - Affected lines: 12
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 52

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 53
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 53

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 13

---

## Issues for scripts/training/fix_notebook_json.py

### Critical Issues (2 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 1 instances.
  - Affected lines: 8
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 54

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 55
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 55

### Minor Issues (2 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 9

---

## Issues for scripts/training/fix_preprocessing_in_notebook.py

### Critical Issues (2 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 1 instances.
  - Affected lines: 12
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 141

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 142
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 142

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 13

---

## Issues for scripts/training/fix_training_arguments.py

### Critical Issues (2 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 1 instances.
  - Affected lines: 12
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 57

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 58
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 58

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 13

---

## Issues for scripts/training/fixed_focal_training.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 14

### Minor Issues (1 total)
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 2 instances.
  - Affected lines: 39, 55

---

## Issues for scripts/training/fixed_training_with_optimized_config.py

### Critical Issues (1 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances.
  - Affected lines: 31

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/training/focal_loss_training.py

### Critical Issues (1 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances.
  - Affected lines: 20

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/training/focal_loss_training_fixed.py

### Critical Issues (3 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 25
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 39
- **Rule: PYL-E0602** (Undefined name detected) - Total count: 1 instances.
  - Affected lines: 184

### Major Issues (2 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 39
- **Rule: PYL-W0613** (Function contains unused argument) - Total count: 1 instances.
  - Affected lines: 99

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 85

---

## Issues for scripts/training/focal_loss_training_robust.py

### Critical Issues (1 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 1 instances.
  - Affected lines: 19

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 43

---

## Issues for scripts/training/focal_loss_training_simple.py

### Critical Issues (1 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 1 instances.
  - Affected lines: 18

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 42

---

## Issues for scripts/training/full_dataset_focal_training.py

### Critical Issues (1 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 1 instances.
  - Affected lines: 18

### Major Issues (1 total)
- **Rule: FLK-W505** (Doc line too long) - Total count: 1 instances.
  - Affected lines: 5

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 42

---

## Issues for scripts/training/full_focal_training.py

### Critical Issues (1 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 1 instances.
  - Affected lines: 19

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 43

---

## Issues for scripts/training/full_scale_focal_training.py

### Critical Issues (1 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 1 instances.
  - Affected lines: 19

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 43

---

## Issues for scripts/training/improve_expanded_training_notebook.py

### Critical Issues (2 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 1 instances.
  - Affected lines: 10
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 122

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 123
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 123

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 11

---

## Issues for scripts/training/minimal_working_training.py

### Critical Issues (1 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances.
  - Affected lines: 15

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/training/monitor_training.py

### Critical Issues (4 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 16
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 6 instances.
  - Affected lines: 37, 50, 106, 166, 200, 228
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 268
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 28

### Major Issues (3 total)
- **Rule: PY-W0070** (Appending to list immediately following its definition) - Total count: 1 instances.
  - Affected lines: 108
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 28
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 176

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/training/pre_training_validation.py

### Critical Issues (1 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances.
  - Affected lines: 28

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/training/restart_training_debug.py

### Critical Issues (1 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances.
  - Affected lines: 3

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/training/robust_domain_adaptation_training.py

### Critical Issues (3 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 11 instances.
  - Affected lines: 25, 67, 96, 126, 140, 151, 182, 219, 242, 296, 324
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 362
- **Rule: PYL-E0602** (Undefined name detected) - Total count: 1 instances.
  - Affected lines: 232

### Major Issues (5 total)
- **Rule: PYL-W1510** (Subprocess run with ignored non-zero exit) - Total count: 5 instances.
  - Affected lines: 42, 48, 54, 59, 104
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 363
- **Rule: FLK-W505** (Doc line too long) - Total count: 3 instances.
  - Affected lines: 5, 8, 352
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 3 instances.
  - Affected lines: 43, 60, 363
- **Rule: PYL-W0612** (Unused variable found) - Total count: 2 instances.
  - Affected lines: 330, 349

### Minor Issues (7 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 2 instances.
  - Affected lines: 105, 235
- **Rule: PYL-R1710** (Inconsistent return statements) - Total count: 1 instances.
  - Affected lines: 182
- **Rule: PYL-R1728** (Redundant list comprehension can be replaced using generator) - Total count: 2 instances.
  - Affected lines: 167, 168
- **Rule: BAN-B602** (Detected subprocess `popen` call with shell equals `True`) - Total count: 1 instances.
  - Affected lines: 104
- **Rule: BAN-B607** (Audit: Starting a process with a partial executable path) - Total count: 4 instances.
  - Affected lines: 42, 48, 54, 59
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 1 instances.
  - Affected lines: 277
- **Rule: PTC-W6004** (Audit required: External control of file name or path) - Total count: 1 instances.
  - Affected lines: 143

---

## Issues for scripts/training/setup_colab_environment.py

### Critical Issues (1 total)
- **Rule: FLK-E128** (Continuation line under-indented for visual indent) - Total count: 1 instances.
  - Affected lines: 69

### Major Issues (3 total)
- **Rule: PYL-W1510** (Subprocess run with ignored non-zero exit) - Total count: 1 instances.
  - Affected lines: 227
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 291
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 3 instances.
  - Affected lines: 39, 68, 291

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 3 instances.
  - Affected lines: 23, 85, 234

---

## Issues for scripts/training/setup_gpu_training.py

### Critical Issues (2 total)
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 22
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 34

### Major Issues (3 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 34
- **Rule: FLK-W505** (Doc line too long) - Total count: 1 instances.
  - Affected lines: 41
- **Rule: PYL-W0511** (Use of `FIXME`/`XXX`/`TODO` encountered) - Total count: 1 instances.
  - Affected lines: 19

### Minor Issues (2 total)
- **Rule: PTC-W0051** (Branches of the `if` statement have similar implementation) - Total count: 1 instances.
  - Affected lines: 104
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 1 instances.
  - Affected lines: 191

---

## Issues for scripts/training/simple_working_training.py

### Critical Issues (1 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances.
  - Affected lines: 17

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/training/summarize_comprehensive_notebook.py

### Critical Issues (2 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 1 instances.
  - Affected lines: 12
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 109

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 110
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 110

### Minor Issues (2 total)
- **Rule: PTC-W0027** (`f-string` used without any expression) - Total count: 2 instances.
  - Affected lines: 27, 104
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 13

---

## Issues for scripts/training/summarize_ultimate_notebook.py

### Critical Issues (2 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 1 instances.
  - Affected lines: 11
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 95

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 96
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 96

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 12

---

## Issues for scripts/training/test_quick_training.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 29

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 3 instances.
  - Affected lines: 98, 155, 181

---

## Issues for scripts/training/validate_improved_notebook.py

### Critical Issues (2 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 1 instances.
  - Affected lines: 9
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 129

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 130
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 130

### Minor Issues (3 total)
- **Rule: PTC-W0027** (`f-string` used without any expression) - Total count: 1 instances.
  - Affected lines: 111
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 10
- **Rule: PY-R1000** (Function with cyclomatic complexity higher than threshold) - Total count: 1 instances.
  - Affected lines: 9

---

## Issues for scripts/training/vertex_automl_training.py

### Critical Issues (3 total)
- **Rule: FLK-E116** (Unexpected indentation in comments) - Total count: 11 instances.
  - Affected lines: 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
- **Rule: FLK-E265** (Block comment should start with `# `) - Total count: 1 instances.
  - Affected lines: 20
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 31

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 31

### Minor Issues (1 total)
- **Rule: PYL-R1723** (Unnecessary `else` / `elif` used after `break`) - Total count: 1 instances.
  - Affected lines: 138

---

## Issues for scripts/training/working_training_script.py

### Critical Issues (1 total)
- **Rule: FLK-E999** (Invalid syntax) - Total count: 1 instances.
  - Affected lines: 9

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for scripts/validation/check_dependencies.py

### Critical Issues (2 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 2 instances.
  - Affected lines: 14, 123
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 139

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 140
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 140

### Minor Issues (3 total)
- **Rule: PTC-W0027** (`f-string` used without any expression) - Total count: 1 instances.
  - Affected lines: 107
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 129
- **Rule: PTC-W6004** (Audit required: External control of file name or path) - Total count: 1 instances.
  - Affected lines: 81

---

## Issues for scripts/validation/validate_security_config.py

### Critical Issues (2 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 2 instances.
  - Affected lines: 14, 242
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 256

### Major Issues (2 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 257
- **Rule: FLK-W291** (Trailing whitespace detected) - Total count: 1 instances.
  - Affected lines: 257

### Minor Issues (1 total)
- **Rule: PTC-W0027** (`f-string` used without any expression) - Total count: 1 instances.
  - Affected lines: 222

---

## Issues for src/api_rate_limiter.py

### Critical Issues (1 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 1 instances.
  - Affected lines: 21

### Major Issues (1 total)
- **Rule: PYL-W0108** (Unnecessary lambda expression) - Total count: 1 instances.
  - Affected lines: 165

### Minor Issues (1 total)
- **Rule: FLK-D204** (1 blank line required after class docstring) - Total count: 1 instances.
  - Affected lines: 23

---

## Issues for src/common/env.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: FLK-W391** (Multiple blank lines detected at end of the file) - Total count: 1 instances.
  - Affected lines: 18

### Minor Issues (0 total)
- None identified.

---

## Issues for src/data/database.py

### Critical Issues (2 total)
- **Rule: FLK-E116** (Unexpected indentation in comments) - Total count: 2 instances.
  - Affected lines: 1, 2
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 19

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 19

### Minor Issues (0 total)
- None identified.

---

## Issues for src/data/embeddings.py

### Critical Issues (2 total)
- **Rule: FLK-E116** (Unexpected indentation in comments) - Total count: 3 instances.
  - Affected lines: 1, 2, 3
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 17

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for src/data/feature_engineering.py

### Critical Issues (2 total)
- **Rule: FLK-E116** (Unexpected indentation in comments) - Total count: 31 instances.
  - Affected lines: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 46

### Major Issues (0 total)
- None identified.

### Minor Issues (0 total)
- None identified.

---

## Issues for src/data/loaders.py

### Critical Issues (1 total)
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 13

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PTC-W6004** (Audit required: External control of file name or path) - Total count: 1 instances.
  - Affected lines: 88

---

## Issues for src/data/models.py

### Critical Issues (0 total)
- None identified.

### Major Issues (2 total)
- **Rule: PYL-W0107** (Unnecessary `pass` statement) - Total count: 1 instances.
  - Affected lines: 30
- **Rule: FLK-W505** (Doc line too long) - Total count: 1 instances.
  - Affected lines: 135

### Minor Issues (0 total)
- None identified.

---

## Issues for src/data/pipeline.py

### Critical Issues (0 total)
- None identified.

### Major Issues (2 total)
- **Rule: FLK-W505** (Doc line too long) - Total count: 3 instances.
  - Affected lines: 49, 82, 166
- **Rule: PYL-W0612** (Unused variable found) - Total count: 3 instances.
  - Affected lines: 183, 184, 226

### Minor Issues (0 total)
- None identified.

---

## Issues for src/data/prisma_client.py

### Critical Issues (3 total)
- **Rule: FLK-E116** (Unexpected indentation in comments) - Total count: 5 instances.
  - Affected lines: 1, 2, 3, 4, 5
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 1 instances.
  - Affected lines: 21
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 14

### Major Issues (2 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 14
- **Rule: FLK-W505** (Doc line too long) - Total count: 1 instances.
  - Affected lines: 24

### Minor Issues (2 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 179
- **Rule: BAN-B607** (Audit: Starting a process with a partial executable path) - Total count: 1 instances.
  - Affected lines: 65

---

## Issues for src/data/sample_data.py

### Critical Issues (3 total)
- **Rule: FLK-E116** (Unexpected indentation in comments) - Total count: 9 instances.
  - Affected lines: 1, 2, 3, 4, 5, 6, 7, 8, 9
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 1 instances.
  - Affected lines: 153
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 25

### Major Issues (1 total)
- **Rule: PYL-W0621** (Re-defined variable from outer scope) - Total count: 3 instances.
  - Affected lines: 195, 215, 247

### Minor Issues (1 total)
- **Rule: PTC-W6004** (Audit required: External control of file name or path) - Total count: 1 instances.
  - Affected lines: 246

---

## Issues for src/data/validation.py

### Critical Issues (1 total)
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 9

### Major Issues (1 total)
- **Rule: FLK-W505** (Doc line too long) - Total count: 1 instances.
  - Affected lines: 156

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 232

---

## Issues for src/input_sanitizer.py

### Critical Issues (1 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 2 instances.
  - Affected lines: 17, 31

### Major Issues (0 total)
- None identified.

### Minor Issues (3 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 154
- **Rule: FLK-D204** (1 blank line required after class docstring) - Total count: 1 instances.
  - Affected lines: 19
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 2 instances.
  - Affected lines: 149, 319

---

## Issues for src/models/emotion_detection/api_demo.py

### Critical Issues (0 total)
- None identified.

### Major Issues (3 total)
- **Rule: PYL-W0706** (Except handler raises immediately) - Total count: 1 instances.
  - Affected lines: 381
- **Rule: PYL-W0603** (`global` statement detected) - Total count: 1 instances.
  - Affected lines: 148
- **Rule: PYL-W0613** (Function contains unused argument) - Total count: 1 instances.
  - Affected lines: 246

### Minor Issues (1 total)
- **Rule: PY-D0002** (Missing class docstring) - Total count: 1 instances.
  - Affected lines: 73

---

## Issues for src/models/emotion_detection/bert_classifier.py

### Critical Issues (0 total)
- None identified.

### Major Issues (2 total)
- **Rule: FLK-W505** (Doc line too long) - Total count: 1 instances.
  - Affected lines: 209
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 251

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 319

---

## Issues for src/models/emotion_detection/dataset_loader.py

### Critical Issues (1 total)
- **Rule: FLK-E402** (Module level import not at the top of the file) - Total count: 1 instances.
  - Affected lines: 30

### Major Issues (2 total)
- **Rule: PY-W2000** (Imported name is not used anywhere in the module) - Total count: 1 instances.
  - Affected lines: 30
- **Rule: PYL-W1203** (Formatted string passed to logging module) - Total count: 4 instances.
  - Affected lines: 225, 252, 283, 340

### Minor Issues (0 total)
- None identified.

---

## Issues for src/models/emotion_detection/hf_loader.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (4 total)
- **Rule: PY-D0002** (Missing class docstring) - Total count: 2 instances.
  - Affected lines: 18, 61
- **Rule: BAN-B108** (Hardcoded temporary directory detected) - Total count: 2 instances.
  - Affected lines: 176, 189
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 4 instances.
  - Affected lines: 24, 66, 109, 130
- **Rule: PY-R1000** (Function with cyclomatic complexity higher than threshold) - Total count: 1 instances.
  - Affected lines: 136

---

## Issues for src/models/emotion_detection/labels.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: FLK-W292** (No newline at end of file) - Total count: 1 instances.
  - Affected lines: 36

### Minor Issues (0 total)
- None identified.

---

## Issues for src/models/emotion_detection/training_pipeline.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (3 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 749
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 1 instances.
  - Affected lines: 748
- **Rule: PY-R1000** (Function with cyclomatic complexity higher than threshold) - Total count: 1 instances.
  - Affected lines: 713

---

## Issues for src/models/secure_loader/integrity_checker.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W1203** (Formatted string passed to logging module) - Total count: 11 instances.
  - Affected lines: 61, 81, 96, 100, 114, 138, 163, 167, 185, 192, 197

### Minor Issues (1 total)
- **Rule: PTC-W6004** (Audit required: External control of file name or path) - Total count: 2 instances.
  - Affected lines: 76, 130

---

## Issues for src/models/secure_loader/model_validator.py

### Critical Issues (1 total)
- **Rule: FLK-E128** (Continuation line under-indented for visual indent) - Total count: 4 instances.
  - Affected lines: 326, 327, 328, 329

### Major Issues (4 total)
- **Rule: PYL-W0404** (Multiple imports for an import name detected) - Total count: 1 instances.
  - Affected lines: 239
- **Rule: PYL-W0621** (Re-defined variable from outer scope) - Total count: 1 instances.
  - Affected lines: 239
- **Rule: FLK-W505** (Doc line too long) - Total count: 1 instances.
  - Affected lines: 389
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 248

### Minor Issues (0 total)
- None identified.

---

## Issues for src/models/secure_loader/sandbox_executor.py

### Critical Issues (0 total)
- None identified.

### Major Issues (4 total)
- **Rule: PYL-W0122** (Audit required: Use of `exec`) - Total count: 1 instances.
  - Affected lines: 171
- **Rule: FLK-W505** (Doc line too long) - Total count: 1 instances.
  - Affected lines: 127
- **Rule: PYL-W0612** (Unused variable found) - Total count: 2 instances.
  - Affected lines: 154, 245
- **Rule: PYL-W1203** (Formatted string passed to logging module) - Total count: 6 instances.
  - Affected lines: 91, 94, 142, 182, 215, 283

### Minor Issues (2 total)
- **Rule: FLK-D204** (1 blank line required after class docstring) - Total count: 1 instances.
  - Affected lines: 20
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 4 instances.
  - Affected lines: 31, 156, 196, 227

---

## Issues for src/models/secure_loader/secure_model_loader.py

### Critical Issues (1 total)
- **Rule: FLK-E128** (Continuation line under-indented for visual indent) - Total count: 10 instances.
  - Affected lines: 208, 209, 210, 211, 212, 331, 332, 333, 334, 335

### Major Issues (2 total)
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 197
- **Rule: PYL-W1203** (Formatted string passed to logging module) - Total count: 8 instances.
  - Affected lines: 105, 106, 152, 255, 266, 279, 314, 327

### Minor Issues (0 total)
- None identified.

---

## Issues for src/models/summarization/api_demo.py

### Critical Issues (0 total)
- None identified.

### Major Issues (4 total)
- **Rule: PYL-W0621** (Re-defined variable from outer scope) - Total count: 1 instances.
  - Affected lines: 36
- **Rule: PYL-W0603** (`global` statement detected) - Total count: 1 instances.
  - Affected lines: 38
- **Rule: PYL-W1203** (Formatted string passed to logging module) - Total count: 3 instances.
  - Affected lines: 51, 52, 55
- **Rule: PYL-W0613** (Function contains unused argument) - Total count: 1 instances.
  - Affected lines: 36

### Minor Issues (1 total)
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 4 instances.
  - Affected lines: 84, 100, 261, 277

---

## Issues for src/models/summarization/dataset_loader.py

### Critical Issues (1 total)
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 7

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 7

### Minor Issues (0 total)
- None identified.

---

## Issues for src/models/summarization/training_pipeline.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 4

### Minor Issues (0 total)
- None identified.

---

## Issues for src/models/voice_processing/__init__.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 6

### Minor Issues (0 total)
- None identified.

---

## Issues for src/models/voice_processing/api_demo.py

### Critical Issues (2 total)
- **Rule: FLK-E116** (Unexpected indentation in comments) - Total count: 25 instances.
  - Affected lines: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 50

### Major Issues (5 total)
- **Rule: PYL-W0706** (Except handler raises immediately) - Total count: 2 instances.
  - Affected lines: 214, 337
- **Rule: PYL-W0621** (Re-defined variable from outer scope) - Total count: 1 instances.
  - Affected lines: 57
- **Rule: PYL-W0603** (`global` statement detected) - Total count: 1 instances.
  - Affected lines: 59
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 269
- **Rule: PYL-W0613** (Function contains unused argument) - Total count: 1 instances.
  - Affected lines: 57

### Minor Issues (2 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 406
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 2 instances.
  - Affected lines: 451, 461

---

## Issues for src/models/voice_processing/audio_preprocessor.py

### Critical Issues (2 total)
- **Rule: FLK-E501** (Line too long) - Total count: 1 instances.
  - Affected lines: 105
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 11

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 11

### Minor Issues (0 total)
- None identified.

---

## Issues for src/models/voice_processing/transcription_api.py

### Critical Issues (3 total)
- **Rule: FLK-E116** (Unexpected indentation in comments) - Total count: 13 instances.
  - Affected lines: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
- **Rule: FLK-E501** (Line too long) - Total count: 5 instances.
  - Affected lines: 43, 161, 233, 234, 255
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 31

### Major Issues (2 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 21
- **Rule: PYL-W1203** (Formatted string passed to logging module) - Total count: 6 instances.
  - Affected lines: 52, 65, 69, 132, 186, 220

### Minor Issues (0 total)
- None identified.

---

## Issues for src/models/voice_processing/whisper_transcriber.py

### Critical Issues (3 total)
- **Rule: FLK-E501** (Line too long) - Total count: 12 instances.
  - Affected lines: 108, 156, 185, 203, 240, 260, 290, 361, 383, 442, 454, 457
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 2 instances.
  - Affected lines: 33, 472
- **Rule: PYL-E1205** (Logging format string contains too many arguments) - Total count: 1 instances.
  - Affected lines: 469

### Major Issues (2 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 16
- **Rule: PYL-W1203** (Formatted string passed to logging module) - Total count: 6 instances.
  - Affected lines: 215, 321, 327, 338, 357, 360

### Minor Issues (1 total)
- **Rule: PYL-R1705** (Unnecessary `else` / `elif` used after `return`) - Total count: 1 instances.
  - Affected lines: 419

---

## Issues for src/monitoring/dashboard.py

### Critical Issues (4 total)
- **Rule: FLK-E129** (Visually indented line with same indent as next logical line) - Total count: 2 instances.
  - Affected lines: 273, 280
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 4 instances.
  - Affected lines: 35, 47, 59, 69
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 1 instances.
  - Affected lines: 369
- **Rule: FLK-E501** (Line too long) - Total count: 8 instances.
  - Affected lines: 138, 155, 188, 192, 211, 260, 312, 329

### Major Issues (2 total)
- **Rule: PY-W2000** (Imported name is not used anywhere in the module) - Total count: 3 instances.
  - Affected lines: 12, 13, 16
- **Rule: PYL-W1203** (Formatted string passed to logging module) - Total count: 1 instances.
  - Affected lines: 135

### Minor Issues (1 total)
- **Rule: FLK-D204** (1 blank line required after class docstring) - Total count: 3 instances.
  - Affected lines: 37, 49, 61

---

## Issues for src/security/jwt_manager.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W1203** (Formatted string passed to logging module) - Total count: 3 instances.
  - Affected lines: 106, 109, 112

### Minor Issues (0 total)
- None identified.

---

## Issues for src/security_headers.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W1203** (Formatted string passed to logging module) - Total count: 9 instances.
  - Affected lines: 79, 282, 284, 377, 384, 391, 398, 489, 518

### Minor Issues (2 total)
- **Rule: PYL-R0201** (Consider decorating method with `@staticmethod`) - Total count: 4 instances.
  - Affected lines: 219, 252, 286, 496
- **Rule: PY-R1000** (Function with cyclomatic complexity higher than threshold) - Total count: 1 instances.
  - Affected lines: 286

---

## Issues for src/unified_ai_api.py

### Critical Issues (5 total)
- **Rule: FLK-E302** (Expected 2 blank lines) - Total count: 21 instances.
  - Affected lines: 70, 122, 175, 204, 327, 334, 343, 355, 370, 907, 947, 1018, 1022, 1063, 1097, 1245, 1645, 1743, 1825, 1945, 2006
- **Rule: FLK-E306** (Expected 1 blank line before a nested definition) - Total count: 1 instances.
  - Affected lines: 83
- **Rule: FLK-E305** (Expected 2 blank lines after end of function or class) - Total count: 2 instances.
  - Affected lines: 197, 324
- **Rule: FLK-E501** (Line too long) - Total count: 11 instances.
  - Affected lines: 410, 1901, 2025, 2030, 2039, 2043, 2044, 2053, 2074, 2103, 2124
- **Rule: FLK-E722** (Do not use bare `except`, specify exception instead) - Total count: 1 instances.
  - Affected lines: 1941

### Major Issues (4 total)
- **Rule: PYL-W0706** (Except handler raises immediately) - Total count: 4 instances.
  - Affected lines: 1054, 1345, 1498, 1815
- **Rule: PYL-W0603** (`global` statement detected) - Total count: 1 instances.
  - Affected lines: 392
- **Rule: PYL-W0612** (Unused variable found) - Total count: 4 instances.
  - Affected lines: 1619, 1806, 1881, 2029
- **Rule: PYL-W0613** (Function contains unused argument) - Total count: 1 instances.
  - Affected lines: 1257

### Minor Issues (5 total)
- **Rule: PY-D0002** (Missing class docstring) - Total count: 1 instances.
  - Affected lines: 768
- **Rule: FLK-D204** (1 blank line required after class docstring) - Total count: 6 instances.
  - Affected lines: 328, 335, 344, 1019, 1121, 1128
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 3 instances.
  - Affected lines: 78, 83, 372
- **Rule: BAN-B104** (Audit: Binding to all interfaces detected with hardcoded values) - Total count: 1 instances.
  - Affected lines: 2158
- **Rule: PY-R1000** (Function with cyclomatic complexity higher than threshold) - Total count: 3 instances.
  - Affected lines: 1366, 1513, 1826

---

## Issues for tests/conftest.py

### Critical Issues (3 total)
- **Rule: FLK-E116** (Unexpected indentation in comments) - Total count: 1 instances.
  - Affected lines: 1
- **Rule: FLK-E501** (Line too long) - Total count: 3 instances.
  - Affected lines: 42, 80, 89
- **Rule: FLK-E303** (Too many blank lines found) - Total count: 1 instances.
  - Affected lines: 16

### Major Issues (2 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 16
- **Rule: PYL-W0613** (Function contains unused argument) - Total count: 1 instances.
  - Affected lines: 125

### Minor Issues (1 total)
- **Rule: FLK-D202** (No blank lines allowed after function docstring) - Total count: 1 instances.
  - Affected lines: 51

---

## Issues for tests/e2e/test_complete_workflows.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0612** (Unused variable found) - Total count: 1 instances.
  - Affected lines: 60

### Minor Issues (0 total)
- None identified.

---

## Issues for tests/integration/test_api_endpoints.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 28

### Minor Issues (2 total)
- **Rule: PYL-R0201** (Consider decorating method with `@staticmethod`) - Total count: 2 instances.
  - Affected lines: 178, 187
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 1 instances.
  - Affected lines: 158

---

## Issues for tests/integration/test_priority1_features.py

### Critical Issues (0 total)
- None identified.

### Major Issues (2 total)
- **Rule: PY-W2000** (Imported name is not used anywhere in the module) - Total count: 3 instances.
  - Affected lines: 12, 13, 21
- **Rule: PYL-W0107** (Unnecessary `pass` statement) - Total count: 2 instances.
  - Affected lines: 485, 491

### Minor Issues (3 total)
- **Rule: PYL-R0201** (Consider decorating method with `@staticmethod`) - Total count: 29 instances.
  - Affected lines: 75, 93, 107, 125, 130, 150, 155, 496, 517, 534, 627, 673, 702, 734, 740, 753, 769, 782, 801, 814, 832, 848, 855, 883, 905, 931, 950, 981, 993
- **Rule: PY-D0002** (Missing class docstring) - Total count: 1 instances.
  - Affected lines: 31
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 1 instances.
  - Affected lines: 378

---

## Issues for tests/unit/test_anomaly_detection.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0404** (Multiple imports for an import name detected) - Total count: 1 instances.
  - Affected lines: 237

### Minor Issues (0 total)
- None identified.

---

## Issues for tests/unit/test_api_models.py

### Critical Issues (0 total)
- None identified.

### Major Issues (2 total)
- **Rule: PYL-W0105** (Unassigned string statement) - Total count: 1 instances.
  - Affected lines: 20
- **Rule: PYL-W0511** (Use of `FIXME`/`XXX`/`TODO` encountered) - Total count: 1 instances.
  - Affected lines: 3

### Minor Issues (1 total)
- **Rule: PYL-R0201** (Consider decorating method with `@staticmethod`) - Total count: 10 instances.
  - Affected lines: 28, 37, 47, 62, 86, 97, 108, 119, 132, 150

---

## Issues for tests/unit/test_api_rate_limiter.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (2 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: PYL-R0201** (Consider decorating method with `@staticmethod`) - Total count: 6 instances.
  - Affected lines: 17, 25, 36, 45, 56, 79

---

## Issues for tests/unit/test_api_security.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0612** (Unused variable found) - Total count: 12 instances.
  - Affected lines: 52, 71, 86, 87, 112, 122, 126, 136, 323, 421, 444, 448

### Minor Issues (0 total)
- None identified.

---

## Issues for tests/unit/test_data_models.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PYL-R0201** (Consider decorating method with `@staticmethod`) - Total count: 13 instances.
  - Affected lines: 24, 32, 42, 63, 74, 101, 111, 130, 142, 165, 175, 200, 206

---

## Issues for tests/unit/test_database.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W1203** (Formatted string passed to logging module) - Total count: 1 instances.
  - Affected lines: 83

### Minor Issues (1 total)
- **Rule: PYL-R0201** (Consider decorating method with `@staticmethod`) - Total count: 12 instances.
  - Affected lines: 23, 29, 33, 38, 43, 47, 56, 66, 70, 76, 89, 93

---

## Issues for tests/unit/test_emotion_detection.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (3 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: PYL-R0201** (Consider decorating method with `@staticmethod`) - Total count: 2 instances.
  - Affected lines: 90, 173
- **Rule: PTC-W0063** (Unguarded next inside generator) - Total count: 2 instances.
  - Affected lines: 137, 142

---

## Issues for tests/unit/test_http_exception_handler.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 3 instances.
  - Affected lines: 10, 24, 38

---

## Issues for tests/unit/test_jwt_manager_extra.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (2 total)
- **Rule: PY-D0002** (Missing class docstring) - Total count: 1 instances.
  - Affected lines: 54
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 6 instances.
  - Affected lines: 10, 29, 34, 56, 66, 91

---

## Issues for tests/unit/test_permission_checker_override.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (1 total)
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 2 instances.
  - Affected lines: 8, 32

---

## Issues for tests/unit/test_sandbox_executor.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0612** (Unused variable found) - Total count: 5 instances.
  - Affected lines: 64, 105, 117, 124, 176

### Minor Issues (1 total)
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 6 instances.
  - Affected lines: 61, 93, 115, 143, 159, 170

---

## Issues for tests/unit/test_secure_model_loader.py

### Critical Issues (0 total)
- None identified.

### Major Issues (3 total)
- **Rule: PYL-W0107** (Unnecessary `pass` statement) - Total count: 1 instances.
  - Affected lines: 54
- **Rule: PYL-W0404** (Multiple imports for an import name detected) - Total count: 2 instances.
  - Affected lines: 271, 385
- **Rule: PYL-W0612** (Unused variable found) - Total count: 5 instances.
  - Affected lines: 174, 229, 310, 319, 435

### Minor Issues (2 total)
- **Rule: FLK-D204** (1 blank line required after class docstring) - Total count: 1 instances.
  - Affected lines: 53
- **Rule: PY-D0003** (Missing module/function docstring) - Total count: 10 instances.
  - Affected lines: 35, 47, 60, 72, 130, 139, 183, 247, 275, 389

---

## Issues for tests/unit/test_validation.py

### Critical Issues (0 total)
- None identified.

### Major Issues (0 total)
- None identified.

### Minor Issues (2 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 2
- **Rule: PYL-R0201** (Consider decorating method with `@staticmethod`) - Total count: 12 instances.
  - Affected lines: 14, 23, 41, 64, 80, 114, 121, 128, 134, 141, 148, 155

---

## Issues for tests/unit/test_validation_enhanced.py

### Critical Issues (0 total)
- None identified.

### Major Issues (1 total)
- **Rule: PYL-W0404** (Multiple imports for an import name detected) - Total count: 1 instances.
  - Affected lines: 111

### Minor Issues (2 total)
- **Rule: FLK-D200** (One-line docstring should fit on one line with quotes) - Total count: 1 instances.
  - Affected lines: 1
- **Rule: PYL-R0201** (Consider decorating method with `@staticmethod`) - Total count: 6 instances.
  - Affected lines: 151, 159, 167, 176, 183, 198

---

**Audit Summary:** Total issues: 2015 (Critical: 1060, Major: 358, Minor: 597).

C0301 not found in the data.

**Discrepancy Note:** The JSON data in DS_AUDIT2.md contains 2015 total issues, which does not match the website's reported 592 issues. Additionally, C0301 (expected 189 instances) is not present in the data.
