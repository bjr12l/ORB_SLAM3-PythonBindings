FROM orb-slam-3:dev

COPY . /home/user/ORB_SLAM3-PythonBindings

WORKDIR ./home/user/ORB_SLAM3-PythonBindings

RUN cd /home/user/ORB_SLAM3-PythonBindings && \
	mkdir build && cd bulid && \
	cmake .. &&\
	make -j$(nproc) &&\
	make install
