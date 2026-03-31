.PHONY: release debug install sync lock smoke smoke-min clean

CMAKE_PREFIX_PATH ?= $(CONDA_PREFIX)

release:
	cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release $(if $(CMAKE_PREFIX_PATH),-DCMAKE_PREFIX_PATH=$(CMAKE_PREFIX_PATH),)
	cmake --build build/release

debug:
	cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug $(if $(CMAKE_PREFIX_PATH),-DCMAKE_PREFIX_PATH=$(CMAKE_PREFIX_PATH),)
	cmake --build build/debug

install:
	python -m pip install --no-build-isolation --editable .

sync:
	uv sync --reinstall-package pygpg

lock:
	uv lock

smoke:
	uv run try.py

smoke-min:
	uv run python -c "import numpy as np; from pygpg.sk import GPGRegressor; rng = np.random.default_rng(42); X = rng.normal(size=(32, 3)).astype(float); y = X[:, 0] + X[:, 1]; model = GPGRegressor(e=200, t=1, g=5, d=2, verbose=False, random_state=42); model.fit(X, y); pred = model.predict(X[:5]); print('fit ok', model.model is not None, pred.shape)"


clean:
	rm -rf build
