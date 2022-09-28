release: BUILDTYPE=release
debug: BUILDTYPE=debug
release: main-build
debug: main-build

main-build:
	mkdir -p build/$(BUILDTYPE)
	cd build/$(BUILDTYPE) && \
	cmake -S ../../ -B . -DCMAKE_BUILD_TYPE=$(BUILDTYPE) && \
	make && \
	cp -r ../../src/swig/python/pygpg . && \
	cp ../../src/swig/python/setup.py . && \
	cp pyface.py pygpg && \
	cp _pyface.* pygpg/ && \
	python setup.py install --user --force && \
	rm -r pygpg pygpg.egg-info && \
	rm setup.py


clean:
	rm -rf build
