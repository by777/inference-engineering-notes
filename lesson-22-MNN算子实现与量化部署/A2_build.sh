#!/bin/bash
g++ -std=c++11 A3_run.cpp -I../mnn-src/include -L../mnn-src/build -lMNN -o A3_run
LD_LIBRARY_PATH=../mnn-src/build ./A3_run