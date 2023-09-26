FROM python:3.11.2 AS base
ENV EMSDK=/opt/emsdk EMSDK_NODE=/opt/emsdk/node/14.18.2_64bit/bin/node \
    PATH=/opt/emsdk:/opt/emsdk/upstream/emscripten:/opt/emsdk/node/14.18.2_64bit/bin:$PATH
RUN git clone https://github.com/emscripten-core/emsdk.git $EMSDK &&\
    emsdk install 3.1.32 && emsdk activate 3.1.32 && pip install pyodide-build==0.23.4

RUN git clone https://github.com/szabolcsdombi/zengl -b 1.13.0 /zengl
RUN pyodide build /zengl -o /web/

RUN git clone --recurse-submodules https://github.com/Zuzu-Typ/PyGLM -b 2.7.0 /pyglm
RUN pyodide build /pyglm -o /web/

COPY assets.pickle /web/

COPY web/webgl /webgl
RUN pyodide build /webgl -o /web/

COPY web/webapp /webapp
RUN pyodide build /webapp -o /web/

WORKDIR /web/
COPY web/index.html web/webgl.js web/main.py /web/
CMD python -m http.server --bind 0.0.0.0 8000
