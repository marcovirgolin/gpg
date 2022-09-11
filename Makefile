release:
	mkdir -p build/release
	cd build/release && \
	cmake -S ../../ -B . -DCMAKE_BUILD_TYPE=release && \
	make && \
	cp -r ../../src/swig/python/pyminigpg . && \
	cp ../../src/swig/python/setup.py . && \
	mv pyface.py pyminigpg && \
	mv _pyface.* pyminigpg/ && \
	python setup.py install --user --force

debug:
	mkdir -p build/debug
	cd build/debug && \
	cmake -S ../../ -B . -DCMAKE_BUILD_TYPE=debug && \
	make && \
	cp -r ../../src/swig/python/pyminigpg . && \
	cp ../../src/swig/python/setup.py . && \
	mv pyface.py pyminigpg && \
	mv _pyface.* pyminigpg/ && \
	python setup.py install --user --force

valgrind:
	mkdir -p build/valgrind
	cd build/valgrind && \
	cmake -S ../../ -B . -DCMAKE_BUILD_TYPE=valgrind && \
	make

clean:
	rm -rf build