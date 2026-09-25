# Dependency and model license inventory

Checked on 2026-09-25 against the packages actually importable by `.venv/bin/python` and the exact versions pinned in `code/business_entity_resolution/requirements.txt`. Evidence comes from installed distribution metadata and installed license/notice files; the links below point to the publishers' repositories. This inventory summarizes declarations and does not replace the upstream license texts.

| Package | Installed/pinned version | Main project license | Official source |
|---|---:|---|---|
| NumPy | 2.1.3 | BSD-3-Clause | [Versioned license](https://github.com/numpy/numpy/blob/v2.1.3/LICENSE.txt) |
| SciPy | 1.16.1 | BSD-3-Clause | [Versioned license](https://github.com/scipy/scipy/blob/v1.16.1/LICENSE.txt) |
| pandas | 2.2.3 | BSD-3-Clause | [Versioned license](https://github.com/pandas-dev/pandas/blob/v2.2.3/LICENSE) |
| scikit-learn | 1.7.2 | BSD-3-Clause | [Versioned license](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/COPYING) |
| XGBoost | 3.1.2 | Apache-2.0 | [Official project license declaration](https://github.com/dmlc/xgboost/blob/master/README.md?plain=1) |
| RapidFuzz | 3.14.1 | MIT | [Official project](https://github.com/rapidfuzz/RapidFuzz) |
| sparse-dot-topn | 1.2.0 | Apache-2.0 | [Official project](https://github.com/ing-bank/sparse_dot_topn) |
| joblib | 1.5.2 | BSD-3-Clause | [Versioned license](https://github.com/joblib/joblib/blob/1.5.2/LICENSE.txt) |
| threadpoolctl | 3.6.0 | BSD-3-Clause | [Versioned license](https://github.com/joblib/threadpoolctl/blob/3.6.0/LICENSE) |
| psutil | 7.2.1 | BSD-3-Clause | [Official project](https://github.com/giampaolo/psutil) |
| pytest | 8.3.4 | MIT | [Versioned license](https://github.com/pytest-dev/pytest/blob/8.3.4/LICENSE) |

The planned trained scorer uses XGBoost, whose installed 3.1.2 metadata declares Apache-2.0. It is trained from the supplied business-matching data; no pretrained model weights or external business records are introduced by these modules. Supporting libraries have their own licenses, as listed above. The parameter/tree count of the actual saved model must be reported from that artifact, rather than inferred from a library's license or version.

Local evidence inspected:

- `numpy-2.1.3.dist-info/LICENSE.txt`
- `scipy-1.16.1.dist-info/LICENSE.txt`
- `pandas-2.2.3.dist-info/LICENSE`
- `scikit_learn-1.7.2.dist-info/licenses/COPYING` and metadata `License-Expression: BSD-3-Clause`
- `xgboost-3.1.2.dist-info/METADATA`, declaring `License: Apache-2.0`; this installed distribution did not list a standalone LICENSE file
- `rapidfuzz-3.14.1.dist-info/licenses/LICENSE` and metadata `License-Expression: MIT`
- `sparse_dot_topn-1.2.0.dist-info/licenses/LICENSE` and `NOTICE`
- `joblib-1.5.2.dist-info/licenses/LICENSE.txt`
- `threadpoolctl-3.6.0.dist-info/licenses/LICENSE`
- `psutil-7.2.1.dist-info/LICENSE`
- `pytest-8.3.4.dist-info/LICENSE`

The NumPy and SciPy wheel notices also enumerate third-party numerical/runtime components. These include OpenBLAS and LAPACK under BSD variants, GCC runtime components under GPL with the GCC runtime exception, and libquadmath under LGPL-2.1-or-later. NumPy's notice additionally lists MIT, Zlib, Apache-2.0 and BSD components. Thus the main project license alone is not a complete description of a redistributed binary wheel. The sparse-dot-topn NOTICE likewise contains pass-through component notices. Preserve the original notices if redistributing those binaries; the source submission can retain pinned dependencies without copying the installed environment.

Versioned GitHub license pages for XGBoost 3.1.2, RapidFuzz 3.14.1, sparse-dot-topn 1.2.0 and psutil 7.2.1 did not load through the browsing tool during this check. Their exact-version declarations above were read from local installed metadata/license files; the official unversioned project links provide corroboration rather than a claim that a versioned page was successfully retrieved.
