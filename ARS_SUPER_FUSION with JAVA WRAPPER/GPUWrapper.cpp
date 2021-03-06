/*MIT License
*
*Copyright (c) 2018 Alysson Ribeiro da Silva
*
*Permission is hereby granted, free of charge, to any person obtaining a copy 
*of this software and associated documentation files (the "Software"), to deal 
*in the Software without restriction, including *without limitation the rights 
*to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
*copies of the Software, and to permit persons to whom the Software is furnished 
*to do so, subject *to the following conditions:
*
*The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*
*THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
*EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
*FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. *IN NO EVENT SHALL THE AUTHORS 
*OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN 
*AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH 
*THE *SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <jni.h>
#include <stdio.h>
#include <string.h>
#include "GPUWrapper.h"

#define DEBUG_JAVA

int getNeuronCount();
void loadConfig(const char* filename);
void _loadNetwork(const char* fileName);
void _saveNetwork(const char* fileName);
void setTraining(bool boolean);
void call(double* stimulus_arr, int stimulus_size, double** output_arr, int* arr_size);

void print(JNIEnv* env, const char* msg){
	// Get system class
	jclass syscls = env->FindClass("java/lang/System");
	// Lookup the "out" field
	jfieldID fid = env->GetStaticFieldID(syscls, "out", "Ljava/io/PrintStream;");
	jobject out = env->GetStaticObjectField(syscls, fid);
	// Get PrintStream class
	jclass pscls = env->FindClass("java/io/PrintStream");
	// Lookup printLn(String)
	jmethodID mid = env->GetMethodID(pscls, "print", "(Ljava/lang/String;)V");
	// Invoke the method
	//jchar *cppstr = // make an array of jchar (UTF-16 unsigned short encoding)
	jstring str = env->NewStringUTF(msg);
	env->CallVoidMethod(out, mid, str);
}

void debug(const char *fmt, ...) {
#ifdef DEBUG_JAVA
	JavaVM* jvm;
	jsize vmnb;
	JNI_GetCreatedJavaVMs(&jvm, 1, &vmnb);

	JNIEnv* env;
	jvm->GetEnv((void**)&env, JNI_VERSION_1_8);
#endif

	va_list arg;
/* Write the error message */
	char buff[64 * 1024];
	va_start(arg, fmt);
	vsprintf(buff, fmt, arg);
	va_end(arg);

#ifdef DEBUG_JAVA
	print(env, buff);
#else
	printf("%s", buff);
#endif
}

// Implementation of native method sayHello() of HelloJNI class
JNIEXPORT jdoubleArray JNICALL Java_ars_simulator_facade_GPUWrapper_prediction(JNIEnv *env, jobject thisObj, jdoubleArray darray) {
	jdouble* doubleArray = env->GetDoubleArrayElements(darray, false);
	int lenght = env->GetArrayLength(darray);

	double* output;
	int arrSize;

	//print(env, "estou aqui\n");

	call(doubleArray, lenght, &output, &arrSize);
	//debug(env, "n... %d", arrSize);

	env->ReleaseDoubleArrayElements(darray, doubleArray, 0);
	jdoubleArray response = env->NewDoubleArray(arrSize);
	env->SetDoubleArrayRegion(response, 0, arrSize, output);
	return response;
}

// Implementation of native method sayHello() of HelloJNI class
JNIEXPORT void JNICALL Java_ars_simulator_facade_GPUWrapper_loadConfig(JNIEnv *env, jobject thisObj, jstring file) {
	// Step 1: Convert the JNI String (jstring) into C-String (char*)
	const char *filename = env->GetStringUTFChars(file, false);
	//debug(env, "n... %s\n", filename);
	loadConfig(filename);
	env->ReleaseStringUTFChars(file, filename);
}

// Implementation of native method sayHello() of HelloJNI class
JNIEXPORT void JNICALL Java_ars_simulator_facade_GPUWrapper_loadNetwork(JNIEnv *env, jobject thisObj, jstring file) {
	// Step 1: Convert the JNI String (jstring) into C-String (char*)
	const char *filename = env->GetStringUTFChars(file, false);
	//debug(env, "n... %s\n", filename);
	_loadNetwork(filename);
	env->ReleaseStringUTFChars(file, filename);
}

JNIEXPORT void JNICALL Java_ars_simulator_facade_GPUWrapper_saveNetwork(JNIEnv *env, jobject thisObj, jstring file) {
	// Step 1: Convert the JNI String (jstring) into C-String (char*)
	const char *filename = env->GetStringUTFChars(file, false);
	//debug(env, "n... %s\n", filename);
	_saveNetwork(filename);
	env->ReleaseStringUTFChars(file, filename);
}

JNIEXPORT void JNICALL Java_ars_simulator_facade_GPUWrapper_setLearning(JNIEnv *env, jobject thisObj, jboolean boolean){
	setTraining(boolean);
}

JNIEXPORT int JNICALL Java_ars_simulator_facade_GPUWrapper_getNeuronCount(JNIEnv *env, jobject thisObj){
	jint ret = getNeuronCount();
	return ret;
}
