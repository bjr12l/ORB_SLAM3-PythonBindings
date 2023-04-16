FROM orb-slam3:dev

RUN apt-get update && \
	apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.8 python3.8-dev python3-pip && \
	python3.8 -m pip install Cython && \
	python3.8 -m pip install numpy

RUN cd /opt && \
    wget https://boostorg.jfrog.io/artifactory/main/release/1.81.0/source/boost_1_81_0.tar.gz && \
    tar -xvzf boost_1_81_0.tar.gz && \
    cd boost_1_81_0 && \
    ./bootstrap.sh --with-libraries=python --with-python=/usr/bin/python3.8 && \
    ./b2 && ./b2 headers && \
    cp -r boost /usr/local/include/ && \
    cp -r stage/lib/* /usr/local/lib/

COPY . /opt/ORB_SLAM3-PythonBinding

# RUN echo "Getting ORB-SLAM3 PythonBindings installation ready ..." && \
# 	cd /opt/ORB_SLAM3-PythonBindings/ && \
# 	mkdir build && cd bulid && \
# 	cmake -D PYTHON_EXECUTABLE=/usr/bin/python3.8 -D ORB_SLAM3_DIR=/usr .. && \
# 	make -j$(nproc) &&\
# 	make install
