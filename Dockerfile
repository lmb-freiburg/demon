FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

RUN apt update && apt install -y python3-pip    \
                                 libhdf5-dev    \
                                 libopencv-dev  \
                                 python3-tk     \
                                 cmake          \
                                 gcc-4.8        \
                                 g++-4.8        \
                                 cmake          \
                                 x11-apps

RUN python3 -m pip install numpy==1.12.1            \
                           tensorflow-gpu===1.4.0   \
                           pillow==2.0.0            \
                           pyparsing===2.1.4        \
                           cycler===0.10.0          \
                           matplotlib===2.1.2
ADD . /home/demon
RUN mkdir /home/demon/lmbspecialops/build
WORKDIR /home/demon/lmbspecialops/build
ENV CC=/usr/bin/gcc-4.8
ENV CXX=/usr/bin/g++-4.8
RUN cmake -DCMAKE_BUILD_TYPE=Release ..
RUN make
ENV PYTHONPATH=/home/demon/lmbspecialops/python
ENV LMBSPECIALOPS_LIB=/home/demon/lmbspecialops/build/lib/lmbspecialops.so
WORKDIR /home/demon/examples
CMD ["python3", "example.py"]
