#!/bin/bash
g++ -std=c++11 A4_trace.cpp -I../mnn-src/include -L../mnn-src/build -lMNN -o A4_trace
LD_LIBRARY_PATH=../mnn-src/build ./A4_trace
