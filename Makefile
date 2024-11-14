CXX = g++
CFLAGS = -Iinclude -I/usr/include/opencv4 -I./lib/onnxruntime/include -g
INC = include
LFLAGS_ONNX = -L./lib/onnxruntime/lib -lonnxruntime -Wl,-rpath=./lib/onnxruntime/lib
LFLAGS_OPENCV = -lopencv_highgui -lopencv_videoio -lopencv_imgproc -lopencv_core -lopencv_imgcodecs
TARGET = build
SRC = src
OBJ = obj

all: directories $(TARGET)/onnx_inference

directories:
	mkdir -p $(OBJ) $(TARGET) data
	 
$(TARGET)/onnx_inference: $(OBJ)/onnx_inference.o $(OBJ)/main.o
	$(CXX) $(CFLAGS) $^ -o $@ $(LFLAGS_ONNX) $(LFLAGS_OPENCV)


$(OBJ)/onnx_inference.o: $(SRC)/onnx_inference.cpp  $(INC)/*
	$(CXX) $(CFLAGS) -c $< -o $@

$(OBJ)/main.o: $(SRC)/main.cpp $(INC)/*
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ)/* $(TARGET)/*
