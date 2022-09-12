release: BUILDTYPE=release
debug: BUILDTYPE=debug
release: main-build
debug: main-build

main-build:
	mkdir -p build/$(BUILDTYPE)
	cd build/$(BUILDTYPE) && \
	cmake -S ../../ -B . -DCMAKE_BUILD_TYPE=$(BUILDTYPE) && \
	make && \
	cp -r ../../src/swig/python/pyminigpg . && \
	cp ../../src/swig/python/setup.py . && \
	cp pyface.py pyminigpg && \
	cp _pyface.* pyminigpg/ && \
	python setup.py install --user --force && \
	rm -r pyminigpg pyminigpg.egg-info && \
	rm setup.py


clean:
	rm -rf build
