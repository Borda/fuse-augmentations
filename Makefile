# Local mirror of the CI extras matrix.
#
# CI installs exactly one backend extra per leg (.github/workflows/ci_testing.yml), so a test that
# imports albumentations, kornia or torchvision without guarding it passes locally -- where every
# backend is installed -- and only fails once CI runs. These targets build the same one-backend
# environments locally so that failure is reachable before a push.
#
#   make test-matrix          # every leg CI runs
#   make test-albumentations  # one leg
#   make test-oldest          # all extras at their minimum pinned versions
#   make clean-envs           # remove the generated environments
#
# Environments are cached under .venv-ci-<extra>; uv reuses its download cache, so a rebuild after
# the first run costs seconds. Delete one to pick up a dependency change.

UV ?= uv
PYTHON_VERSION ?= 3.10
EXTRAS ?= albumentations kornia torchvision all
ENV_PREFIX ?= .venv-ci

# CI installs a CPU torch build; a CUDA wheel here would change nothing about import guards and
# costs gigabytes.
export UV_TORCH_BACKEND = cpu

.PHONY: test-matrix test-oldest clean-envs

test-matrix: $(addprefix test-,$(EXTRAS)) test-oldest

# One explicit rule per extra rather than a `test-%` pattern rule: GNU make does not apply pattern
# rules to targets declared .PHONY, so the pattern form silently reports "nothing to be done".
#
# The documentation tests under tests/integration/ are generated, and CI generates them only on the
# `all` leg -- every other leg never sees them. Locally they persist on disk from a previous `all`
# run, so a single-backend leg has to ignore them or it reports failures CI cannot produce.
define TEST_LEG
.PHONY: test-$(1)
test-$(1):
	@echo "=== extras=$(1) (python $$(PYTHON_VERSION))"
	@$$(UV) venv "$$(ENV_PREFIX)-$(1)" --python $$(PYTHON_VERSION)
	@VIRTUAL_ENV="$$(ENV_PREFIX)-$(1)" $$(UV) pip install -e ".[$(1)]" --group dev
ifeq ($(1),all)
	@"$$(ENV_PREFIX)-$(1)/bin/python" .github/scripts/generate_doc_tests.py
	@"$$(ENV_PREFIX)-$(1)/bin/python" -m pytest tests/ -q
else
	@"$$(ENV_PREFIX)-$(1)/bin/python" -m pytest tests/ -q --ignore=tests/integration
endif
endef

$(foreach extra,$(EXTRAS),$(eval $(call TEST_LEG,$(extra))))

# The `oldest` leg, which pins every `>=` requirement to its floor. It is the leg that catches a test
# written against behaviour a newer dependency added -- torch 2.2 has no half-precision CPU
# `grid_sample`, and its resampling does not agree with a current build to the last intensity level.
#
# min_deps.py rewrites pyproject.toml in place, which CI can do because its checkout is disposable.
# Here the original is restored whether the install succeeds or fails, and the tests then run against
# the live source tree so a fix can be tried without committing it first.
test-oldest:
	@echo "=== extras=all, minimum pins (python $(PYTHON_VERSION))"
	@$(UV) venv "$(ENV_PREFIX)-oldest" --python $(PYTHON_VERSION)
	@cp pyproject.toml "$(ENV_PREFIX)-oldest/pyproject.orig"
	@VIRTUAL_ENV="$(ENV_PREFIX)-oldest" $(UV) pip install tomlkit
	@"$(ENV_PREFIX)-oldest/bin/python" .github/scripts/min_deps.py; \
	  VIRTUAL_ENV="$(ENV_PREFIX)-oldest" $(UV) pip install -e ".[all]" --group dev; \
	  status=$$?; \
	  cp "$(ENV_PREFIX)-oldest/pyproject.orig" pyproject.toml; \
	  exit $$status
	@"$(ENV_PREFIX)-oldest/bin/python" .github/scripts/generate_doc_tests.py
	@"$(ENV_PREFIX)-oldest/bin/python" -m pytest tests/ -q

clean-envs:
	rm -rf $(addprefix $(ENV_PREFIX)-,$(EXTRAS)) "$(ENV_PREFIX)-oldest"
