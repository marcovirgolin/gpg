
release:
	mkdir -p build/release
	cd build/release && \
	cmake -S ../../ -B . -DCMAKE_BUILD_TYPE=release && \
	make

debug:
	mkdir -p build/debug
	cd build/debug && \
	cmake -S ../../ -B . -DCMAKE_BUILD_TYPE=debug && \
	make

valgrind:
	mkdir -p build/valgrind
	cd build/valgrind && \
	cmake -S ../../ -B . -DCMAKE_BUILD_TYPE=valgrind && \
	make

clean:
	rm -rf build