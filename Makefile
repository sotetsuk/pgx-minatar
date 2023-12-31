.PHONY: install-dev clean install test


install-dev:
	python3 -m pip install \
		pytest==7.1.2 \
		matplotlib \
		ipython \
		jax[cpu] \
		dm-haiku

clean:
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -name "*pycache*" | xargs rm -rf

install:
	python3 -m pip install --upgrade pip setuptools
	python3 -m pip install .

test:
	python3 -m pytest --doctest-modules tests/test_*.py
